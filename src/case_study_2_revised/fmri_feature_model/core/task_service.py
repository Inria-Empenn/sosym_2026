class TaskService:

    def get_task_contrasts(self, name: str):
        match name.lower():
            case "auditory":
                return self.get_auditory_contrasts()
            case "motor":
                return self.get_motor_contrasts()

    def get_auditory_contrasts(self):
        return [
        ('listening > rest', 'T', ['listening'], [1])
    ]

    def get_motor_contrasts(self):
        return []