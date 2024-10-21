import pickle


class Tokeniser:
    def __init__(self, lookup_pkl="utils/lookup.pkl") -> None:
        with open(lookup_pkl, "rb") as f:
            word_to_id, id_to_word = pickle.load(f)
        self.word_to_id = word_to_id
        self.id_to_word = id_to_word
        self.default_token = 0

    def tokenise(self, text: str) -> list[int]:
        split = self._preprocess(text)
        tokens = [
            self.word_to_id[word] if word in self.word_to_id else self.default_token
            for word in split
        ]
        return tokens

    def _preprocess(self, text: str) -> list[str]:
        text = text.lower()
        text = text.replace(".", " <PERIOD> ")
        text = text.replace(",", " <COMMA> ")
        text = text.replace('"', " <QUOTATION_MARK> ")
        text = text.replace(";", " <SEMICOLON> ")
        text = text.replace("!", " <EXCLAMATION_MARK> ")
        text = text.replace("?", " <QUESTION_MARK> ")
        text = text.replace("(", " <LEFT_PAREN> ")
        text = text.replace(")", " <RIGHT_PAREN> ")
        text = text.replace("--", " <HYPHENS> ")
        text = text.replace("?", " <QUESTION_MARK> ")
        text = text.replace(":", " <COLON> ")
        return text.split()


if __name__ == "__main__":
    tokeniser = Tokeniser("lookup.pkl")

    print(tokeniser.tokenise("King Queen the"))
