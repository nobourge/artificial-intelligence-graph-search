import sys
import inspect


class AutoIndent(object):
    def __init__(self, stream):
        self.stream = stream
        self.offset = 0
        self.frame_cache = {}

    def flush(self):
        pass

    def indent_level(self):
        i = 0
        base = sys._getframe(2)
        f = base.f_back
        while f:
            if id(f) in self.frame_cache:
                i += 1
            f = f.f_back
        if i == 0:
            # clear out the frame cache
            self.frame_cache = {id(base): True}
        else:
            self.frame_cache[id(base)] = True
        return i

    def write(self, stuff):
        indentation = "  " * self.indent_level()

        def indent(l):
            if l:
                return indentation + l
            else:
                return l

        stuff = "\n".join([indent(line) for line in stuff.split("\n")])
        self.stream.write(stuff)
        # write stuff in file ./src/log.txt
        with open("./src/log.txt", "a") as f:
            # flush log.txt
            f.seek(0)
            f.write(stuff)

