import argparse
import asyncio
import logging
from wyoming.info import Info, TtsVoice
from wyoming.server import AsyncTcpServer
from wyoming_f5_tts.f5tts_event_handler import F5TTSProcessManager, F5TTSEventHandler

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="tcp://0.0.0.0:10200", help="URI to listen on")
    parser.add_argument("--streaming", action="store_true", default=True, help="Enable text streaming")
    parser.add_argument("--samples-per-chunk", type=int, default=1024, help="Audio samples per chunk")
    parser.add_argument("--chunk-timeout", type=float, default=0.5, help="Timeout for partial sentence synthesis")
    parser.add_argument("--auto-punctuation", type=str, default=".,!?", help="Punctuation to append")
    parser.add_argument("--model-name", default="F5-TTS/F5-TTS", help="F5-TTS model name")
    parser.add_argument("--ref-audio", default=None, help="Path to reference audio for voice cloning")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER = logging.getLogger(__name__)

    process_manager = F5TTSProcessManager(model_name=args.model_name)
    await process_manager.initialize()

    wyoming_info = Info(
        tts=[
            TtsVoice(name="F5-TTS", description="F5-TTS English", languages=["en"]),
            TtsVoice(name="F5-TTS-ZH", description="F5-TTS Chinese", languages=["zh"]),
        ]
    )

    _LOGGER.info("Starting Wyoming server on %s", args.uri)
    server = AsyncTcpServer(args.uri)
    await server.run(lambda: F5TTSEventHandler(wyoming_info, args, process_manager))

if __name__ == "__main__":
    asyncio.run(main())

