
class FileService:
    """
    FileService is a class that provides functionality to manage and read files 
    using a mapping of file names to their full paths.
    """
    def __init__(self):
        # Initialize a dictionary to map file names to their full paths
        self.file_map = {}
        self.add_file('WeatherSystemPrompt.txt', './app/prompts/WeatherSystemPrompt.txt') 
    def add_file(self, file_name, file_path):
        self.file_map[file_name] = file_path

    def read_file(self, file_name):
        if file_name in self.file_map:
            file_path = self.file_map[file_name]
            try:
                with open(file_path, 'r') as file:
                    content = file.read()
                return content
            except FileNotFoundError:
                raise RuntimeError( f"File '{file_name}' not found at path '{file_path}'.")
        else:
            raise RuntimeError(f"File '{file_name}' not found in the file map.")
