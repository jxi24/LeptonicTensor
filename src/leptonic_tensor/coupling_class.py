import ufo_grammer

# class CouplingInfo():
#     def __init__(self, model, ufo_coupling):
#         self.name = ufo_coupling.name
#         self.model = model
#         ufo = ufo_grammer.UFOParser(self.model)
#         coupling = ufo("{} := {}".format(self.name, ufo_coupling.value))
#         self.value = ufo(self.name)
#         # except:
#         #     ufo(self.name)
#         #     self.value = None
        

class Coupling():
    def __init__(self, model, ufo_coupling):
        self.model = model
        self.name = ufo_coupling.name
        self.value = self.model.coupling_map[self.name]
        
    def __str__(self):
        return '{}: {}'.format(
            self.name, self.value)  