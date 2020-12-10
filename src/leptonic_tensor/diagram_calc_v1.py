import json

def anti(particle, part_to_anti):
    return part_to_anti[particle]

def partial3_checker(partialSol, part_Dict):
    if len(partialSol) > 3:
        return False
    else:
        antiPart = anti(part_Dict[partialSol[0]], part_to_anti)
        part1, part2 = part_Dict[partialSol[1]], part_Dict[partialSol[2]]
        for vertex in vertices:
            if antiPart in vertex:
                new_vertex = list(set(vertex) - {antiPart})
                if part1 in new_vertex:
                    newParticle = list(set(new_vertex) - {part1})[0]
                    if part2 == newParticle:
                        return True
        return False

def constructCandidates(partialSol, part_Dict):
    """
    Iterates through partialSol[1:] and checks if antiparticle of partialSol[0] has a vertex
    with possible candidates. We assume partialSol[0] is incoming, so we use antiparticle
    to correctly check vertices. If such a vertex exists, we add candidate and the particle
    resulting from this interaction to the candidate dictionary. Returns candidate dictionary.

    Parameters
    ----------
    partialSol : list
        List of numbered strings representing partial solutions.
    part_Dict : dictionary
        Dictionary matching numbered strings to particles.

    Returns
    -------
    candidates : dictionary
        Dictionary matching candidates to particle resulting from interaction with partialSol[0].

    """
    candidates = {}
    for e in partialSol[1:]:
        for vertex in vertices:
            antiPart = anti(part_Dict[partialSol[0]], part_to_anti)
            if antiPart in vertex:
                new_vertex = list(set(vertex) - {antiPart})
                
                if part_Dict[e] in new_vertex: #Handle repeated particles.
                    #print("partialSol[0]: ({},{}), e: ({},{}), vertex: {}".format(partialSol[0],part_Dict[partialSol[0]],e,part_Dict[e], vertex))
                    
                    newParticle = list(set(new_vertex) - {part_Dict[e]})[0]
                    candidates[e] = newParticle
    return candidates

def makeMove(key, value, partialSol, part_Dict):
    newSol = partialSol
    newSol.remove(key)
    #print("  New solution before sorting: {}".format(newSol))
    newSol.sort(key=len, reverse=True)
    newSol[0] += key
    #print("  New solution after sorting and adding key: {}".format(newSol))
    part_Dict[newSol[0]] = value
    return newSol
    
def unmakeMove(key, value, partialSol, part_Dict):
    oldSol = partialSol + [key]
    #print("  unmakeMove oldSol 1 = {}".format(oldSol))
    oldSol[0] = oldSol[0].replace(key,'')
    #print("  unmakeMove oldSol 2 = {}".format(oldSol))
    oldSol.sort(key=len, reverse=True)
    #print("  unmakeMove oldSol final = {}".format(oldSol))
    return oldSol
    
def backtrack(partialSol, part_Dict, diagrams):
    #print("Partial Solution: {}".format(partialSol))
    if len(partialSol) == 2:
        if part_Dict[partialSol[0]] == part_Dict[partialSol[1]]:
            diagrams.append(partialSol)
        return partialSol
    else:
        if partial3_checker(partialSol, part_Dict):
            key, value = partialSol[1], part_Dict[partialSol[2]]
            #print(partialSol)
            partialSol = makeMove(key, value, partialSol, part_Dict)
            #print(partialSol)
            diagrams.append(partialSol)
            partialSol = unmakeMove(key, value, partialSol, part_Dict)
            #print(partialSol)
            return partialSol
        #print("Partial solution: {}".format(partialSol))
        candidates = constructCandidates(partialSol, part_Dict)
        #print("  Candidates: {}".format(candidates))
        for key in candidates:
            value = candidates[key]
            #print(key, value)
            partialSol = makeMove(key, value, partialSol, part_Dict)
            #print("  (Key, Value): ({},{}) Partial Solution after makeMove: {}".format(key, value, partialSol))
            partialSol = backtrack(partialSol, part_Dict, diagrams)
            partialSol = unmakeMove(key, value, partialSol, part_Dict)
            #print("  (Key, Value): ({},{}) Partial Solution after unmakeMove: {}".format(key, value, partialSol))
        return partialSol

def main(particles, vertices):
    diagrams = []
    partialSol = []
    part_Dict = {}
    for i, p in enumerate(particles, start=1):
        partialSol += str(i)
        part_Dict[str(i)] = p
    partialSol_lists = [partialSol]
    for i in range(1,len(partialSol)):
        newPartial = partialSol[:]
        newPartial[0], newPartial[i] = partialSol[i], partialSol[0]
        partialSol_lists.append(newPartial)
    for e in partialSol_lists:
        backtrack(e, part_Dict, diagrams)
    return diagrams, part_Dict
    
if __name__ == "__main__":
    
    vertices = [["vmu_tilde", "vmu", "Z"], ["vmu_tilde", "mu_minus", "W_plus"], 
            ["vmu", "mu_plus", "W_minus"], ["mu_plus", "mu_minus", "Z"],
            ["W_minus", "W_plus", "Z"]]

        #  ["mu_plus", "mu_minus", "A"] avoid QED for now.
        
    part_to_anti = {"vmu":"vmu_tilde", "vmu_tilde":"vmu", "Z":"Z", 
                    "W_plus":"W_minus", "W_minus":"W_plus", "mu_minus":"mu_plus", 
                    "mu_plus":"mu_minus"}

    particles = ["vmu", "Z", "vmu", "mu_minus", "mu_plus"]
    # Assume first particle is incoming and rest are outgoing.
    
    diagrams, part_Dict = main(particles, vertices)
    
    print("Neutrino trident process with a Z boson:")
    print("Feynman diagrams:\n{}".format(diagrams))
    print("Particle dictionary: {}".format(json.dumps(part_Dict, indent=4)))
    
    vertices2 = [["mu_plus", "mu_minus", "Z"], ["mu_plus", "mu_minus", "gamma"]]
    particles2 = ["mu_plus", "mu_plus", "mu_minus", "mu_plus"]
    
    diagrams2, part_Dict2 = main(particles2, vertices2)
    
    print("Bhabha scattering for muons:")
    print("Feynman diagrams:\n{}".format(diagrams2))
    print("Particle dictionary: {}".format(json.dumps(part_Dict2, indent=4)))