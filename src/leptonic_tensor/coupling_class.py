class Coupling():
    def __init__(self, model, ufo_coupling):
        self.model = model
        self.name = ufo_coupling.name
        self.value = self.model.coupling_map[self.name]
        
    def __str__(self):
        return '{}: {}'.format(
            self.name, self.value)  