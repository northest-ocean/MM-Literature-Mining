class MMBeController:


    def __init__(self):
        self.handlers = dict()

    def register_command(self, commandID, handler):
        self.handlers[commandID] = handler
    
    def unregister_command(self, commandID):
        self.handlers[commandID] = None
    
    def get_command_handler(self, commandID):
        return 