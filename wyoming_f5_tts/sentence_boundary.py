class SentenceBoundaryDetector:
    def __init__(self):
        self.buffer = ""

    def add_chunk(self, text: str) -> list[str]:
        self.buffer += text
        sentences = []
        while any(p in self.buffer for p in ".!?"):
            for p in ".!?":
                if p in self.buffer:
                    idx = self.buffer.index(p) + 1
                    sentence = self.buffer[:idx].strip()
                    if sentence:
                        sentences.append(sentence)
                    self.buffer = self.buffer[idx:]
                    break
        if len(self.buffer) > 100:
            sentences.append(self.buffer)
            self.buffer = ""
        return sentences

    def finish(self) -> str:
        if self.buffer:
            result = self.buffer
            self.buffer = ""
            return result
        return ""

def remove_asterisks(text: str) -> str:
    return text.replace("*", "")

