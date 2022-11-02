# https://gist.github.com/danmelton/183313 adjusted to iphone keyboard
import random

typos = {
    'q': ['w', 'a'],
    'w': ['q', 'a', 's', 'e'],
    'e': ['w', 's', 'd', 'r'],
    'r': ['e', 'd', 'f', 't'],
    't': ['r', 'f', 'g', 'y'],
    'y': ['t', 'g', 'h', 'u'],
    'u': ['y', 'h', 'j', 'i'],
    'i': ['u', 'j', 'k', 'o'],
    'o': ['i', 'k', 'l', 'p'],
    'p': ['o', 'l'],
    'a': ['z', 's', 'w', 'q', 'z'],
    's': ['a', 'z', 'x', 'd', 'e', 'w'],
    'd': ['s', 'x', 'c', 'f', 'r', 'e', 'z'],
    'f': ['d', 'c', 'v', 'g', 't', 'r'],
    'g': ['f', 'v', 'b', 'h', 'y', 't', 'c'],
    'h': ['g', 'b', 'n', 'j', 'u', 'y', 'v'],
    'j': ['h', 'n', 'm', 'k', 'i', 'u', 'b'],
    'k': ['j', 'm', 'l', 'o', 'i', 'n'],
    'l': ['k', 'p', 'o', 'm'],
    'z': ['x', 's', 'a', 's'],
    'x': ['z', 'c', 'd', 's', 'x'],
    'c': ['x', 'v', 'f', 'd', 'g'],
    'v': ['c', 'b', 'g', 'f', 'h'],
    'b': ['v', 'n', 'h', 'g', 'j'],
    'n': ['b', 'm', 'j', 'h', 'k'],
    'm': ['n', 'k', 'j', 'm']
}


def change_char(text: str) -> str:
    l = len(text)
    idx = random.randint(0, l - 1)
    tmp = list(text)
    char = tmp[idx]
    if char not in typos:
        return text
    candidate_idx = random.randint(0, len(typos[char]) - 1)
    new_char = typos[char][candidate_idx]
    tmp[idx] = new_char
    return ''.join(tmp)


def remove_char(text: str) -> str:
    l = len(text)
    idx = random.randint(0, l - 1)
    tmp = list(text)
    del tmp[idx]
    return ''.join(tmp)


def transpose(text: str) -> str:
    l = len(text)
    idx = random.randint(0, l - 2)
    tmp = list(text)
    t = tmp[idx]
    tmp[idx] = tmp[idx + 1]
    tmp[idx + 1] = t
    return ''.join(tmp)


ops = {
    0.5: change_char,
    0.75: remove_char,
    1.0: transpose
}


class TypoGenerator:
    def __init__(self, p: float, seed: int = 42):
        self.p = p
        random.seed(seed)

    def create_typos(self, text: str, debug: bool = False) -> str:
        if self.p == 0.0:
            return text
        splitted = text.split(' ')
        result = []
        for token in splitted:
            if len(token) < 2:
                if debug:
                    print(f"{token} has length less than 2, skipping")
                result.append(token)
                continue
            r = random.random()
            if r > self.p:
                if debug:
                    print(f"{token} will not be modified. Needed r>{p}, was r={r}")
                result.append(token)
                continue
            typo_type = random.random()
            for prob, op in ops.items():
                if typo_type < prob:
                    if debug:
                        print(f"Selected typo type {op} based on r={typo_type} for token={token}")
                    result.append(op(token))
                    break

        res = ' '.join(result)
        if debug:
            print(f"Was '{text}', is '{res}'")
        return res


if __name__ == '__main__':
    typo_generator = TypoGenerator(p=0.3)
    print(typo_generator.create_typos("How to remove an element from a list by index"))
