import re, os, sys
from multiprocessing import current_process
from multiprocessing import Manager, Queue
from itertools import zip_longest

from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from logging import FileHandler, StreamHandler
import logging
import atexit
import warnings

import wcwidth

def get_logger(name, *names, **kwargs):
    if not hasattr(get_logger, "_loggers"):
        get_logger._loggers = {}  # Cache loggers here

    name = ' '.join(names)
    if name not in get_logger._loggers:
        # Initialize and store the logger only when it is first used
        get_logger._loggers[name] = Logger(*names, **kwargs)

    return get_logger._loggers[name]

def add_custom_level(levelName, levelNum, methodName=None):
    """
    Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class. (Stack Overflow)

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present 

    Example
    -------
    >>> logging.addLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE

    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError('{} already defined in logging module'.format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError('{} already defined in logging module'.format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError('{} already defined in logger class'.format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)

try:
    add_custom_level('MESSAGE', logging.INFO - 5)
except AttributeError:
    pass

class LoggerFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    cyan = "\x1b[36;20m"
    purple = "\x1b[35;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    gray = "\x1b[90;20m"
    reset = "\x1b[0m"

    def __init__(self, fmt = None, datefmt = None, style = "%", validate = True, *, defaults = None, **kwargs):
        
        if LoggerFormatter.use_jupyter():
            leveltype = "%(levelname)s"
            color = self.reset
            emoji = ""
        else:
            leveltype = "%(levelicon)s" if kwargs.pop("use_icons", True) else "%(levelname)s"
            emoji = "%(emoji)s"
            color = self.gray

        fmt = fmt or f"{color}{emoji}%(asctime)s %(pid)s%(name)s{self.reset}%(levelcolor)s{leveltype}%(location)s{self.reset}%(message)s"
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)

    @staticmethod
    def use_jupyter():
        try:
            from IPython import get_ipython
            ipy_str = str(type(get_ipython()))
            if 'zmqshell' in ipy_str:
                return True
            if 'terminal' in ipy_str:
                return True
        except:
            pass
            
        return False

    @staticmethod
    def wlen(text):
        """Compute the real display width of a string, accounting for double-width characters."""
        width = 0
        for char in text:
            w = wcwidth.wcwidth(char)  # Handle complex emoji properly
            if w > 0:
                width += w
        return width

    @staticmethod
    def wljust(text, width, fillchar=' '):
        """Left-justify a string based on real display width."""
        current_width = LoggerFormatter.wlen(text)
        padding = max(0, width - current_width)
        return text + fillchar * padding

    @staticmethod
    def wrjust(text, width, fillchar=' '):
        """Right-justify a string based on real display width."""
        current_width = LoggerFormatter.wlen(text)
        padding = max(0, width - current_width)
        return fillchar * padding + text

    @staticmethod
    def wcenter(text, width, fillchar=' '):
        """Center a string based on real display width."""
        current_width = LoggerFormatter.wlen(text)
        padding = max(0, width - current_width)
        left_pad = padding // 2
        right_pad = padding - left_pad
        return fillchar * left_pad + text + fillchar * right_pad

    @staticmethod
    def spacer(text):
        """Replace all visible characters with spaces but keep shell colors and tabs."""
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        ansi_sequences = {match.start(): match.group() for match in ansi_escape.finditer(text)}
        replaced_text = re.sub(r'[^\t\x1B]', " ", text)
        for pos, ansi_seq in ansi_sequences.items():
            replaced_text = replaced_text[:pos] + ansi_seq + replaced_text[pos + len(ansi_seq):]

        return replaced_text

    NAMES = {
        logging.MESSAGE : "       ",
        logging.DEBUG   : "DEBUG",
        logging.INFO    : "INFO",
        logging.WARNING : "WARNING",
        logging.ERROR   : "ERROR",
        logging.CRITICAL: "FATAL"
    }

    ICONS = {
        logging.MESSAGE : "  ",
        logging.DEBUG   : "🔍",
        logging.INFO    : "📢",
        logging.WARNING : "⚠️ ",
        logging.ERROR   : "🚨",
        logging.CRITICAL: "☠️ "
    }

    COLORS = {
        logging.MESSAGE : "",
        logging.DEBUG   : cyan,
        logging.INFO    : green,
        logging.WARNING : yellow,
        logging.ERROR   : red,
        logging.CRITICAL: bold_red
    }
          
        
    def format(self, record):
        
        formatter = self._fmt
        formatter = logging.Formatter(formatter)

        record.pid   = getattr(record, 'pid', "")  # Process ID
        record.endl  = getattr(record, 'endl', '\n')
        record.emoji = getattr(record, 'emoji', None)  # Emoji (default empty)
        if record.emoji is not None and self.wlen(record.emoji) < 4:
            record.emoji = record.emoji[:2]
            record.emoji = "["+self.wcenter(record.emoji,2)+"]" if record.emoji != "" else ""
        record.emoji = self.wcenter(record.emoji[:4] or "", 4)
        
        record.levelicon  = self.ICONS.get(record.levelno, "")
        record.location   = "%(filename)s:%(lineno)d" % record.__dict__ if record.levelno > logging.ERROR else ""
        record.levelname  = self.NAMES.get(record.levelno, "").rjust(max(len(v) for v in self.NAMES.values()))
        for key, value in record.__dict__.items():
            if not isinstance(value, str):
                continue
            if key == "endl":
                continue
 
            if value != "": 
                record.__dict__[key] = value + " "
        
        record.levelcolor = self.COLORS.get(record.levelno, "")

        empty = logging.LogRecord(**record.__dict__, level=logging.DEBUG)
        empty.__dict__.update(record.__dict__)
        empty.levelicon = self.ICONS.get(logging.MESSAGE, "")
        empty.msg = ""

        # Multi-line messages
        message = record.msg.split("\n")
        if len(message) > 1:
            message = [message[0].strip()] + [formatter.format(empty) + " " + line.strip() for line in message[1:] if line.strip() != ""]
            message = [msg for msg in message if msg.strip() != ""]
        
        record.msg = "\n".join(message)
        
        return "\r" + formatter.format(record).strip() + (record.endl if record.levelno < logging.WARNING else "")

class Logger: # not inheriting on purpose (_pickle.PicklingError: logger cannot be pickled)
    
    listener   = None
    logger     = None
    verbose    = None
    quiet      = False
    queue      = None
    shm        = None

    MESSAGE  = logging.MESSAGE 
    DEBUG    = logging.DEBUG   
    INFO     = logging.INFO    
    WARNING  = logging.WARNING 
    ERROR    = logging.ERROR   
    CRITICAL = logging.CRITICAL
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('listener', None)
        state.pop('logger', None)
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        Logger.queue = state['queue']
        Logger.shm = state['shm']
        self.__init__(*self.names, **self.__dict__)

    def __init__(self, *names, **kwargs):

        # Setup verbosity
        self.verbose = kwargs.get("verbose", None)
        if self.verbose is None: 
            self.verbose = os.getenv("VERBOSE", os.getenv("DEBUG", None))
        if self.verbose is None:
            self.verbose = Logger.verbose

        self.emoji = kwargs.get("emoji", "")
        self.quiet = kwargs.get("quiet", None)
        self.show_worker_label = kwargs.get("show_worker_label", True)
        
        if self.verbose:
            self.quiet = False
        if self.quiet is None: 
            self.quiet = os.getenv("QUIET", False)

        # Setup logger
        self.names = list(names)
        self.group = kwargs.get("group", True)
        self.id = f"{current_process().name}#{id(self)}"

        self.logger = logging.getLogger(self.id)
        self.logger.setLevel(kwargs.get("level", None) or logging.DEBUG if self.verbose else logging.MESSAGE)
        self.logger.name = self.id

        # Setup queue
        self.queue = kwargs.get("queue", None)
        if not isinstance(self.queue, type(Queue)):
            self.queue = Logger.queue or Queue()

        # Setup listener & shared resources
        self.shm = kwargs.get("shm", Logger.shm if self.group is not None else None)
        if current_process().name == "MainProcess" and self.group is not None:
            try:
                self.shm = Logger.shm or Manager().dict()
                Logger.shm = self.shm
            except:
                pass

            self._create_listener(*self._console_handler(**kwargs), *self._file_handler(**kwargs))

        # Setup handler
        self._fork_logger()
        atexit.register(self.close)
        self.debug("Logger initialized.")
    
    def close(self):

        try:
            self.debug("Closing logger...")
            if self.listener:
                self.listener.stop()

            if self.queue and Logger.queue != self.queue:
                self.queue.close()
                self.queue.join_thread()

            if self.shm and Logger.shm != self.shm:
                self.shm.clear()
        except:
            pass

    def __del__(self):
        self.close()

    def _formatted_name(self, *args, **kwargs):

        args = self.names if args is [None] else self.names + list(args)
        process_name = current_process().name if current_process().name != "MainProcess" and self.show_worker_label else ""
        if process_name:
            if len(args) > 0: args[0] = str(args[0]) + " #" + process_name     
            else: args = ["#" + process_name]   

        level = kwargs.get("level", self.logger.level) # Shift only if message is printed
        max_lengths = [len(arg or "") for arg in args]
        if self.shm is not None and not self.quiet and self.group is not None and self.logger.level <= level:
            try:
                if not self.id+"-"+str(self.group) in self.shm or self.shm[self.id+"-"+str(self.group)] != max_lengths:
                    self.shm[self.id+"-emoji"] = "++"

                self.shm[self.id+"-"+str(self.group)] = [max(x, y) for x, y in zip_longest(self.shm.get(self.id+"-"+str(self.group), []), max_lengths, fillvalue=0)]
                max_lengths = {key:(value or "") for key, value in self.shm.items() if key.endswith("-"+str(self.group))}
                max_lengths = [max(values) for values in zip_longest(*max_lengths.values(), fillvalue=0)]

            except Exception as e:
                print(e)

        return ' '.join([(arg or "").ljust(max_length if max_length > 0 else len(arg or "")) for arg, max_length in zip_longest(args, max_lengths, fillvalue=0) if arg is not None])

    def is_master(self):
        return current_process().name == "MainProcess"
    def is_worker(self):
        return not self.is_master()
    
    def _create_listener(self, *handlers):

        if self.listener is not None:
            return self.listener
        
        # Queue listener to process logs in the main process
        self.listener = QueueListener(self.queue, *handlers)
        self.listener.start()

        return self.listener
    
    def _fork_logger(self):

        if self.logger is None:
            raise ValueError("Logger not initialized.")
        if any(isinstance(handler, QueueHandler) for handler in self.logger.handlers):
            return self.logger

        self.logger = logging.getLogger(self.id)

        queue_handler = logging.handlers.QueueHandler(self.queue)
        self.logger.addHandler(queue_handler)

        self.logger.propagate = False

        return self.logger

    def _console_handler(self, **kwargs):

        stdout = StreamHandler(sys.stdout)
        stdout.terminator = ""
        stdout.setFormatter(LoggerFormatter(**kwargs))
        stdout.addFilter(self._filter_stdout)
        stdout.addFilter(self._filter_quiet)

        stderr = StreamHandler(sys.stderr)
        stdout.terminator = ""
        stderr.setFormatter(LoggerFormatter(**kwargs))
        stderr.addFilter(self._filter_stderr)
        stderr.addFilter(self._filter_quiet)

        return stdout, stderr

    def _file_handler(self, **kwargs):

        out = kwargs.pop("out", None)
        err = kwargs.pop("err", out)
        maxBytes = kwargs.get("maxBytes", 100 * 1048576) # 100Mio

        common = out == err
        if isinstance(out, str): 
            out = os.path.expanduser(out)
            if os.path.dirname(out):
                os.makedirs(os.path.dirname(out), exist_ok=True)
            out = FileHandler(out) if maxBytes is None else RotatingFileHandler(out)

        if out is not None:
            out.setFormatter(LoggerFormatter(**kwargs))
            if not common: out.addFilter(self._filter_stdout)
            yield out

        if not common:
            if isinstance(err, str): 
                err = os.path.expanduser(err)
                if os.path.dirname(err):
                    os.makedirs(os.path.dirname(err), exist_ok=True)
                err = FileHandler(err) if maxBytes is None else RotatingFileHandler(err, maxBytes=maxBytes)

            if err is not None:
                err.setFormatter(LoggerFormatter(**kwargs))
                err.addFilter(self._filter_stderr)          
                yield err

    def _filter_stdout(self, record): # Filter to allow DEBUG and INFO messages
        return record.levelno < logging.WARNING
    def _filter_stderr(self, record): # Filter to allow WARNING, ERROR, and CRITICAL messages
        return record.levelno >= logging.WARNING
    def _filter_quiet(self, record):
        return not self.quiet

    def get_logger(self):
        return self.logger or logging.getLogger(self.id)

    def get_level(self):
        return self.get_logger().level

    def set_name(self, *names):
        self.names = names

    def log(self, level, msg, *args, **kwargs):

        title = kwargs.pop("title", None)
        self.logger.name = self._formatted_name(title, level=level)

        verbose = self.verbose if self.verbose is not None else Logger.verbose
        if verbose: self.logger.setLevel(logging.DEBUG)

        kwargs["extra"] = kwargs.pop("extra", {})
        kwargs["extra"]["endl"] = kwargs.pop("endl", "\n")
        if level >= logging.WARNING and kwargs["extra"]["endl"] != "\n":
            warnings.warn("Changing newline character is not supported for WARNING messages and higher priorities.")

        kwargs["extra"]['pid'] = kwargs.pop("pid", str(os.getpid()).rjust(5)) # Assumed to be 5 digits max 
        kwargs["extra"]['emoji'] = kwargs.pop("emoji", "")
        if kwargs["extra"]['emoji'] is None and self.shm is not None and self.id+"-emoji" in self.shm:
            try:
                kwargs["extra"]['emoji'] = self.shm[self.id+"-emoji"]
                del self.shm[self.id+"-emoji"]

            except Exception as e:
                print(e)

        if kwargs["extra"]['emoji'] is None:
            kwargs["extra"]['emoji'] = self.emoji

        kwargs["stacklevel"] = kwargs.get("stacklevel", 0) + 3
        self.logger.log(level, self.format(msg, **kwargs), *args, **kwargs)
    
    def debug(self, *args, **kwargs):
        self.log(logging.DEBUG, *args, **kwargs)
    def message(self, *args, **kwargs):
        self.log(logging.MESSAGE, *args, **kwargs)
    def info(self, *args, **kwargs):
        self.log(logging.INFO, *args, **kwargs)
    def exception(self, *args, exc_info = True, **kwargs):
        self.log(logging.ERROR, self.format(*args), exc_info = exc_info, **kwargs)
    def warning(self, *args, **kwargs):
        self.log(logging.WARNING, self.format(*args), **kwargs)
    def error(self, *args, **kwargs):
        self.log(logging.ERROR, self.format(*args), **kwargs)
    def critical(self, *args, **kwargs):
        self.log(logging.CRITICAL, self.format(*args), **kwargs)
        sys.exit(1)  # Ensure the program exits with the error message
    def fatal(self, *args, **kwargs):
        self.log(logging.CRITICAL, self.format(*args), **kwargs)
        sys.exit(1)  # Ensure the program exits with the error message

    def format(self, *args, **kwargs):

        vars = kwargs.get("vars", ["PWD", "HOME", "CONDA_PREFIX", "TMPDIR"])
        msg = ' '.join(map(str, args))

        if isinstance(vars, list):
            for key in vars:
                if key in os.environ:
                    value = os.environ[key]
                    slash = "/" if os.environ[key].endswith("/") else ""
                    msg = msg.replace(value, f"{LoggerFormatter.purple}${key}{LoggerFormatter.reset}{slash}")

        return msg

# Custom warning handler
logger = Logger("Warnings")
def customwarn(message, category, filename, lineno, file=None, line=None):
    logger.warning(f"{LoggerFormatter.yellow}{str(message).capitalize()}{LoggerFormatter.reset}\n{category.__name__} in {filename}:{lineno}")

warnings.showwarning = customwarn
