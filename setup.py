
from setuptools import setup

def requirements():
    reqs = []
    with open('requirements.txt', 'r') as fp:
        for req in fp:
            # remove endline, white space, and anything after '#'
            req = req.rstrip('\n').strip().split('#')[0]
            if req is '':
                continue

            reqs.append(req)

    return reqs

setup(
    name='autograd_hacks',
    version='0.0.2',
    packages=['autograd_hacks'],
    long_description=open('README.md').read(),
    install_requires=requirements()
)
