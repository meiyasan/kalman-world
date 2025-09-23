import os
from argparse import ArgumentParser
from string import Template
from typing import IO, Union

from .logging import Logger

class Usage(ArgumentParser):

    verbose = False
    debug = False
    quiet = False

    def __init__(self, *args, **kwargs):

        # Get `motd` and check if it was explicitly set to `False`
        motd = kwargs.pop("motd", None)
        if motd is False:
            self.motd = []  # Explicitly disabled
        elif isinstance(motd, str):
            self.motd = [motd]  # Convert single string to a list
        else:
            # Use MOTD environment variable if set, otherwise default paths
            self.motd = [os.getenv("MOTD")] if os.getenv("MOTD") else [
                os.path.join(os.path.dirname(__file__), "share/motd"),
                "share/motd",
                "../share/motd",
            ]

        super().__init__(*args, **kwargs)
        self.add_argument('--debug', action='store_true', help='Debug mode')
        self.add_argument('--verbose', help='Verbosity')
        self.add_argument('--quiet', action='store_true', help='Quiet mode')

    def parse_args(self, *args, **kwargs):

        args = super().parse_args(*args, **kwargs)

        if Logger.verbose is None:
            Logger.verbose = None
            if args.verbose:
                Logger.verbose = args.verbose
            if args.debug:
                Logger.verbose = args.debug

        if Logger.quiet is None:
            Logger.quiet = False
            if args.quiet:
                Logger.quiet = args.quiet
            if Usage.quiet:
                Logger.quiet = Usage.quiet
        
        if not Logger.quiet:
            self.print_motd()

        return args

    def substitute_with_template(self, text):
        """Substitute environment variables in the text using string.Template."""
        template = Template(text)
        return template.safe_substitute(os.environ)

    def get_substituted_motd(self, motd_file):
        try:
            # Read the MOTD content
            with open(motd_file, 'r') as file:
                motd_content = file.read()

            # Substitute environment variables
            substituted_content = self.substitute_with_template(motd_content)
            return substituted_content.replace("\\t", "\t").replace("\\n", "\n").replace("\\033", "\033").replace("\\\\", "\\")

        except FileNotFoundError:
            raise RuntimeError(f"The file {motd_file} does not exist")
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")

    def find_existing_motd(self) -> Union[str, None]:
        """Find the first existing MOTD file from the list."""
        return next((path for path in self.motd if os.path.exists(path)), None)

    def print_motd(self) -> None:
        """Print the first found MOTD file."""
        motd_path = self.find_existing_motd()
        if motd_path is not None and os.path.exists(motd_path):

            motd = self.get_substituted_motd(motd_path)
            print(motd)

            self.motd = []

    def print_help(self, file: IO[str] | None = None) -> None:
        self.print_motd()
        return super().print_help(file)

    def error(self, message):
        self.print_motd()
        super().error(message)

# Usage
if __name__ == "__main__":
    parser = Usage(motd='test_motd.txt', description='Test MOTD with ANSI colors')
    parser.print_help()
