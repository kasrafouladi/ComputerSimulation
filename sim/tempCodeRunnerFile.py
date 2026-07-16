     while True:
            busy = sum(1 for i in range(self.p.n_machines) if not self.machine_free[i] and not self.machine_down[i])
            machines_down = sum(self.machine_down)
            available = self.p.n_machines - busy - machines_down
            queue_length = len(self.buffer)
            queue_full = queue_length >= self.p.buffer_capacity
            utilization = busy / self.p.n_machines
            pressure = busy + queue_length
            over_capacity_pressure = pressure > self.p.n_machines

            self.hourly_records.append({
                "time": self.env.now,
                "machines_busy": busy,
                "available_machines": available,
                "queue_length": queue_length,
                "queue_full": queue_full,
                "machine_utilization": utilization,
                "system_pressure": pressure,
                "over_capacity_pressure": over_capacity_pressure,
                "machines_down": machines_down,
            })
            yield self.env.timeout(1.0)