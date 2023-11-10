import os

os.makedirs('cache/repos')
os.makedirs('cache/dataset')

def gather_go_codec():
    """
    git clone https://github.com/ugorji/go go-codec
    cd go-codec/codec
    git checkout XXX
    go mod tidy
    go test
    """
    pass