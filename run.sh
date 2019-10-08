#!/bin/bash
for i in 0 3 5 6 7 8 9
do
docker run -d --name m$i -v "$PWD":/usr/src/myapp -w /usr/src/myapp iedmrc/monte-carlo-vrp:latest -i 'input_thu.json' -e 5000 -p 100000 -g ${i}
done

for i in 1 4 5 6 7 8 9
do
docker run -d --name n$i -v "$PWD":/usr/src/myapp -w /usr/src/myapp iedmrc/monte-carlo-vrp:latest -i 'input_thu.json' -e 5000000 -g ${i}
done