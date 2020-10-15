import numpy as np
import cmath



phi = input('input an angle:    ')
phi = float(phi)

x = cmath.rect(1, phi)
print(f'input was  {phi}')

y = cmath.rect(1, phi + np.pi)

z =  np.angle(y) - np.angle(x)
print(f'z is  {z}')

y2  = y = cmath.rect(1, phi - np.pi)
z2 = np.angle(y2) - np.angle(x)
print(f'z2 is {z2}')

rotation_vector = cmath.rect(1, np.pi)
#z3 = np.angle(np.multiply(x, rotation_vector)) - np.angle(x)
z3 = np.angle(np.divide(np.multiply(x,rotation_vector ), x/2))
print(f'z3 is {z3}')

a = [-6.960716247558594-8.37857723236084j, 16.18398666381836+1.236741065979004j,-0.0065220324-0.009812462j, 0.00652203243225813+0.009812462143599987j,3,4,-5,-6,7,8]

for counter in range(len(a))[1:]:
    print(a[counter])
    amp = np.abs(a[counter])/np.abs(a[counter-1])
    a[counter]  = np.multiply(amp*a[counter-1], cmath.rect(1,np.pi))

    print(np.angle(np.divide(a[counter], a[counter-1])))
