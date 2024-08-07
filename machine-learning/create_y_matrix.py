import re
import glob
import numpy as np


def create_y_matrix():
    # Read the elastool.out file

    y_matrix = []

    constants = glob.glob("finished/constants/*")
    constants.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
    for afile in constants:
        with open(afile, 'r') as file:
            content = file.read()
        file.close()
        # Start with stability check, remove unstable data
        if re.search(r"\bNOT\b", content):
            print("\n!!!WARNING!!! UNSTABLE STRUCTURE\n"+
                   "Please remove {} along with the associated OUTCAR and CONTCAR\n".format(afile))
            #print(afile,outcar,contcar)
            #os.system("Remove-Item {}, {}, {}".format(afile,outcar,contcar))
        else:

            pattern = r'C(\d{2})'
            # Find all occurrences of C{digit} and extract the numerical value
            matches = re.findall(pattern, content)
            # Save the numerical values in a dictionary
            values = []
            for match in matches:
                digit = match
                value_pattern = r'C' + digit + r'\s*=\s*(-?[\d.]+)'
                match_value = re.search(value_pattern, content)
                if match_value:
                    value = float(match_value.group(1))
                    # ignore negative and near-zero values
                    if value > 25.0:
                        values.append(value)

            y_matrix.append(values)


    y_matrix = np.array(y_matrix).reshape(len(constants),len(values))
    np.save("y-matrix.npy",y_matrix)
    print(y_matrix.shape)

create_y_matrix()