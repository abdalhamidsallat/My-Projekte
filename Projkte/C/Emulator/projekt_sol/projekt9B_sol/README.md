# Troubleshooting

## Compiling under Windows doesn't work

As Windows provides a different library for the functions `ntohs` and `htons` 
the changes shown in the file `winsock2.patch` must be applied.
The patch can be applied with:
```
patch -p1 < winsock2.patch
```
or manually if the package manager doesn't provide `patch`



