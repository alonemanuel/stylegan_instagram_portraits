class DataProcessException(Exception):
    def __init__(self, subject):
        self.message = f'Data processing exception, has to to with:\n{subject}'
        super().__init__(self.message)

    def __str__(self):
        return self.message