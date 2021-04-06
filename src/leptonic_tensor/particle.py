class Particle:
    max_id = 0
    model = None

    def __init__(self, i, pid):
        self.id = i
        self.pid = pid
        part_info = Particle.model.particle_map[pid]
        try:
            self.mass = Particle.model.parameter_map[part_info.mass]
        except KeyError:
            self.mass = 0.0
        try:
            self.width = Particle.model.parameter_map[part_info.width]
        except KeyError:
            self.width = 0.0
        self.charge = part_info.charge
        self.spin = part_info.spin

    def __str__(self):
        # sid = self.get_id()
        return f'({self.id}, {self.pid}, {self.mass}, {self.width})'
    
    def conjugate(self):
        pid = self.pid
        part = Particle.model.particle_map[pid]
        if part.name != part.antiname:
            pid = -self.pid
        return Particle(self.id, pid)

    def __repr__(self):
        return str(self)

    def __lt__(self, other):
        sid = self.get_id()
        oid = other.get_id()
        if sid < oid:
            return True
        elif oid < sid:
            return False
        return self.pid < other.pid

    def __hash__(self):
        return self.pid

    def get_id(self):
        if self.id >= Particle.max_id:
            return PARTMAP[self.id]
        return self.id

    def __eq__(self, other):
        return (self.get_id() == other.get_id()
                and abs(self.pid) == abs(other.pid))

    def is_fermion(self):
        if 0 < self.pid < 20:
            return True
        return False

    def is_antifermion(self):
        if -20 < self.pid < 0:
            return True
        return False

    def is_vector(self):
        if (20 < self.pid or self.pid < -20) and self.pid != 25:
            return True
        return False

    def is_scalar(self):
        if self.pid == 25:
            return True
        return False

    def massless(self):
        return self.mass == 0.0
