from rich.console import Console
from rich.pretty import pprint as rich_pprint
from rich.progress import track as rich_track


console = Console()

# rich pprint wrapped to staple it to the global console
def pprint(*args, **kwargs):
    kwargs['console'] = console
    return rich_pprint(*args, **kwargs)

def track(*args, **kwargs):
    kwargs['console'] = console
    return rich_track(*args, **kwargs)
