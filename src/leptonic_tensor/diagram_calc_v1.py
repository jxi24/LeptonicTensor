import sys

sys.setrecursionlimit(2*10**4)

def anti(particle, part_to_anti):
    return part_to_anti[particle]
    
def constructCandidates(partialSol, part_Dict):
    candidates = {}
    for e in partialSol[1:]:
        for vertex in vertices:
            if part_Dict[partialSol[0]] in vertex:
                new_vertex = list(set(vertex) - {part_Dict[partialSol[0]]})
                if part_Dict[e] in new_vertex: #Handle repeated particles.
                    #print("partialSol[0]: ({},{}), e: ({},{}), vertex: {}".format(partialSol[0],part_Dict[partialSol[0]],e,part_Dict[e], vertex))
                    newParticle = list(set(new_vertex) - {part_Dict[e]})[0]
                    candidates[e] = newParticle
    return candidates

def makeMove(key, value, partialSol, part_Dict):
    newSol = list(set(partialSol) - {key})
    newSol.sort()
    newSol[0] += key
    part_Dict[newSol[0]] = value
    return newSol
    
def unmakeMove(key, value, partialSol, part_Dict):
    oldSol = partialSol + [key]
    oldSol[0] = oldSol[0].replace(key,'')
    oldSol.sort()
    return oldSol
    
def backtrack(partialSol, part_Dict, diagrams):
    if len(partialSol) == 2:
        if part_Dict[partialSol[0]] == part_Dict[partialSol[1]]:
            diagrams.append(partialSol)
    else:
        #print("Partial solution: {}".format(partialSol))
        candidates = constructCandidates(partialSol, part_Dict)
        #print("Candidates: {}".format(candidates))
        for key in candidates:
            value = candidates[key]
            #print(key, value)
            partialSol = makeMove(key, value, partialSol, part_Dict)
            #print(partialSol)
            backtrack(partialSol, part_Dict, diagrams)
            partialSol = unmakeMove(key, value, partialSol, part_Dict)

def main(particles, vertices):
    diagrams = []
    partialSol = []
    part_Dict = {}
    for i, p in enumerate(particles, start=1):
        partialSol += str(i)
        part_Dict[str(i)] = p
    backtrack(partialSol, part_Dict, diagrams)
    return diagrams, part_Dict
    
if __name__ == "__main__":
    
    vertices = [["vmu_tilde", "vmu", "Z"], ["vmu_tilde", "mu_minus", "W_plus"], 
            ["vmu", "mu_plus", "W_minus"], ["mu_plus", "mu_minus", "Z"],
            ["W_minus", "W_plus", "Z"]]

        #  ["mu_plus", "mu_minus", "A"] avoid QED for now.
        
    part_to_anti = {"vmu":"vmu_tilde", "vmu_tilde":"vmu", "Z":"Z", 
                    "W_plus":"W_minus", "W_minus":"W_plus", "mu_minus":"mu_plus", 
                    "mu_plus":"mu_minus"}

    particles = ["vmu_tilde", "Z", "vmu", "mu_minus", "mu_plus"]
    
    diagrams, part_Dict = main(particles, vertices)
    
    print("Feynman diagrams: {}".format(diagrams))
    print("Particle dictionary: {}".format(part_Dict))