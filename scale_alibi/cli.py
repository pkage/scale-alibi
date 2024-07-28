import asyncclick as click
from . import console
from dotenv import load_dotenv

# pre-work
load_dotenv()

@click.group()
def cli():
    pass

@cli.command('debug', help='debug hook')
def cli_debug():
    console.log('beginning hooks')


