import argparse
import json
import logging
from typing import Any, Dict, Optional
import asyncio
import torch
from transformers import pipeline
from funasr import AutoModel
import torchaudio
import numpy as np

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from sentence_boundary import SentenceBoundaryDetector, remove_asterisks

_LOGGER = logging.getLogger(__name__)

class F5TTSProcessManager:
    def __init__(self, model_name: str = "SWivid/F5-TTS", sample_rate: int = 24000):
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.phonemizer = None
        self.processes_lock = asyncio.Lock()

    async def initialize(self):
        """Load F5-TTS model and phonemizer."""
        _LOGGER.debug("Loading F5-TTS model: %s", self.model_name)
        self.model = pipeline("text-to-speech", model=self.model_name, device=self.device)
        self.phonemizer = AutoModel(model="paraformer-zh", model_path="damo/speech_paraformer-large_asr_nat-zh-cn-16k-96000h-vocab8404-pytorch")
        # Warm up model with a dummy inference
        await self._generate_audio("init", None)

    async def _generate_audio(self, text: str, ref_audio: Optional[str]) -> bytes:
        """Generate audio stream from text using F5-TTS."""
        try:
            # Convert text to phonemes using funasr
            phoneme_result = self.phonemizer.generate(input=text)
            phonemes = phoneme_result[0]["text"]

            # Generate audio with F5-TTS
            audio = self.model(
                phonemes,
                reference_audio=ref_audio,
                chunk_length=100,  # Adjust for streaming
                stream=True
            )

            # Convert to PCM bytes
            audio_chunks = []
            for chunk in audio:
                waveform = torch.tensor(chunk["audio"], dtype=torch.float32)
                # Resample to target sample rate if needed
                if chunk["sampling_rate"] != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(
                        orig_freq=chunk["sampling_rate"],
                        new_freq=self.sample_rate
                    )(waveform)
                # Convert to 16-bit PCM
                waveform = (waveform * 32767).numpy().astype(np.int16).tobytes()
                audio_chunks.append(waveform)
            return b"".join(audio_chunks)
        except Exception as e:
            _LOGGER.error("Error generating audio: %s", e)
            raise

    async def get_process(self, voice_name: Optional[str] = None) -> "F5TTSProcessManager":
        """Return the process manager itself (single instance)."""
        if self.model is None:
            await self.initialize()
        return self

class F5TTSEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        cli_args: argparse.Namespace,
        process_manager: F5TTSProcessManager,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cli_args = cli_args
        self.wyoming_info_event = wyoming_info.event()
        self.process_manager = process_manager
        self.sbd = SentenceBoundaryDetector()
        self.is_streaming: Optional[bool] = None
        self._synthesize: Optional[Synthesize] = None

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info_event)
            _LOGGER.debug("Sent info")
            return True

        try:
            if Synthesize.is_type(event.type):
                if self.is_streaming:
                    return True
                synthesize = Synthesize.from_event(event)
                synthesize.text = remove_asterisks(synthesize.text)
                return await self._handle_synthesize(synthesize)

            if not self.cli_args.streaming:
                return True

            if SynthesizeStart.is_type(event.type):
                stream_start = SynthesizeStart.from_event(event)
                self.is_streaming = True
                self.sbd = SentenceBoundaryDetector()
                self._synthesize = Synthesize(text="", voice=stream_start.voice)
                _LOGGER.debug("Text stream started: voice=%s", stream_start.voice)
                return True

            if SynthesizeChunk.is_type(event.type):
                assert self._synthesize is not None
                stream_chunk = SynthesizeChunk.from_event(event)
                async with asyncio.timeout(self.cli_args.chunk_timeout):
                    for sentence in self.sbd.add_chunk(stream_chunk.text):
                        _LOGGER.debug("Synthesizing stream sentence: %s", sentence)
                        self._synthesize.text = sentence
                        await self._handle_synthesize(self._synthesize)
                return True

            if SynthesizeStop.is_type(event.type):
                assert self._synthesize is not None
                self._synthesize.text = self.sbd.finish()
                if self._synthesize.text:
                    await self._handle_synthesize(self._synthesize)
                await self.write_event(SynthesizeStopped().event())
                _LOGGER.debug("Text stream stopped")
                return True

            return True
        except asyncio.TimeoutError:
            _LOGGER.error("Timeout processing streaming chunk")
            await self.write_event(Error(text="Timeout in streaming", code="TimeoutError").event())
            return False
        except Exception as err:
            await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
            raise err

    async def _handle_synthesize(self, synthesize: Synthesize) -> bool:
        _LOGGER.debug(synthesize)
        text = " ".join(synthesize.text.strip().splitlines())
        if self.cli_args.auto_punctuation and text and not any(text[-1] == p for p in self.cli_args.auto_punctuation):
            text += self.cli_args.auto_punctuation[0]

        async with self.process_manager.processes_lock:
            voice_name = synthesize.voice.name if synthesize.voice else None
            ref_audio = synthesize.voice.speaker if synthesize.voice else None  # Use speaker as reference audio path
            f5tts_proc = await self.process_manager.get_process(voice_name=voice_name)

            # Stream audio
            rate = f5tts_proc.sample_rate
            width = 2  # 16-bit PCM
            channels = 1  # Mono
            bytes_per_chunk = width * channels * self.cli_args.samples_per_chunk
            buffer_size = bytes_per_chunk * 5  # Buffer 5 chunks (~100ms)

            audio_buffer = bytearray()
            await self.write_event(AudioStart(rate=rate, width=width, channels=channels).event())

            # Generate audio stream
            audio_bytes = await f5tts_proc._generate_audio(text, ref_audio)
            audio_buffer.extend(audio_bytes)

            # Stream chunks
            while len(audio_buffer) >= bytes_per_chunk:
                chunk = audio_buffer[:bytes_per_chunk]
                await self.write_event(AudioChunk(audio=chunk, rate=rate, width=width, channels=channels).event())
                audio_buffer = audio_buffer[bytes_per_chunk:]

            # Send remaining audio
            if audio_buffer:
                await self.write_event(AudioChunk(audio=audio_buffer, rate=rate, width=width, channels=channels).event())

            await self.write_event(AudioStop().event())
            _LOGGER.debug("Completed request")
        return True
