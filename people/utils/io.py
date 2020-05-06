import sys

# Terminal colors
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'


def printc(vmsg, color=''):
    endc = ENDC if color != '' else ''
    sys.stdout.write(color + vmsg + endc)
    sys.stdout.flush()

def printcn(vmsg, color=''):
    endc = ENDC if color != '' else ''
    sys.stdout.write(color + vmsg + endc + '\n')
    sys.stdout.flush()

def warning(vmsg, color=WARNING):
    endc = ENDC if color != '' else ''
    sys.stderr.write(color + vmsg + endc + '\n')
    sys.stderr.flush()

def sprintcn(vmsg, color=''):
    endc = ENDC if color != '' else ''
    return color + vmsg + endc + '\n'

# Aliases.

printnl = printcn
