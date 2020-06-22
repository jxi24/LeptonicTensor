import ufo_grammer

class CouplingInfo():
    def __init__(self, ufo_coupling):
        self.name = ufo_coupling.name
        try:
            self.value = ufo_grammer.ufo(ufo_coupling.value)
        except:
            self.value = ufo_coupling.value
        

class Coupling():
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.value = self.model.coupling_map[self.name]
        
    def __str__(self):
        return '{}: {}'.format(
            self.name, self.value)
        