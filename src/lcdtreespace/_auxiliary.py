import numpy as np

def J(yl,yr,ep):
    if np.abs(yr-yl) > ep:
        return (np.exp(yr) - np.exp(yl))/(yr-yl)
    else:
        ymean = (yl + yr) / 2
        z2 = ((yl-ymean)**2 + (yr-ymean)**2)/2
        z3 = ((yl-ymean)**3 + (yr-ymean)**3)/3
        return np.exp(ymean) * (1 + z2/6 + z3/24)

def J1(yl,yr,ep):
    if np.abs(yr-yl) > ep:
        return np.exp(yl) * (np.exp(yr-yl) - 1 - yr + yl)/(yr-yl)**2
    else:
        return np.exp(yl) * (1/2 + (yr-yl)/6 + (yr-yl)**2/24 + (yr-yl)**3/120)

def J00(yl,yr,ep=1e-10):
    if np.abs(yr-yl) > ep:
        return (np.exp(yr) - np.exp(yl))/(yr-yl)
    else:
        ymean = (yl + yr) / 2
        z2 = ((yl-ymean)**2 + (yr-ymean)**2)/2
        z3 = ((yl-ymean)**3 + (yr-ymean)**3)/3
        return np.exp(ymean) * (1 + z2/6 + z3/24)

def J10(yl,yr,ep=1e-10):
    # ylで微分
    if np.abs(yr-yl) > ep:
        return np.exp(yl) * (np.exp(yr-yl) - 1 - yr + yl)/(yr-yl)**2
    else:
        return np.exp(yl) * (1/2 + (yr-yl)/6 + (yr-yl)**2/24 + (yr-yl)**3/120)

def J20(yl, yr, ep=1e-10):
    y = yr-yl
    if np.abs(yr-yl) > ep:
        return np.exp(yl) * 2 * (np.exp(y) -1 -y -y**2/2)/(y**3)
    else:
        return np.exp(yl) * (1/3 + y/12 + y**2/60 + y**3/360)

def J30(yl, yr, ep=1e-10):
    y = yr-yl
    if np.abs(yr-yl) > ep:
        return np.exp(yl) * 6 * (np.exp(y) -1 -y -y**2/2 - y**3/6)/(y**4)
    else:
        return np.exp(yl) * (1/4 + y/20 + y**2/120 + y**3/840)

def J000(y1, y2, y3, ep=1e-10):
    y_order = np.sort(np.array([y1, y2, y3]))
    if np.abs(y_order[2] - y_order[0]) > ep:
        return (J00(y_order[1], y_order[2]) - J00(y_order[0], y_order[1]))/(y_order[2] - y_order[0])
    else:
        y_mean = y_order.mean()
        return np.exp(y_mean)/2

def J100(y1, y2, y3, ep=1e-10):
    if np.abs(y2-y1) > ep:
        return (J00(y2, y3) - J00(y1, y3) - J10(y1, y3)*(y2-y1))/((y2-y1)**2)
    else:
        return J20(y1, y3)/2 + J30(y1,y3) * (y2-y1)/6
