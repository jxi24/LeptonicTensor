from . import ufo_grammer


class Parameter:
    def __init__(self, ufo_parameter):
        self.name = self.ufo_parameter.name
        self.type = self.ufo_parameter.type
        self.tex = self.ufo_parameter.texname

        if ufo_parameter.nature == "external":
            self.type = "external"
            self.lhacode = self.ufo_parameter.lhacode
            self.lhablock = self.ufo_parameter.lhablock
            if ufo_parameter.lhacode is not None:
                raise ValueError("Invalid parameter: "
                                 "{}, requires lhablock".format(self.name))
        elif ufo_parameter.nature == "internal":
            self.type = "internal"
        else:
            raise ValueError("Invalid parameter: {}, should be internal "
                             "or external".format(self.name))

        self.value = ufo_parameter.value

    def register(self):
        ufo_grammer.ufo(self.name + ":=" + str(self.value))
