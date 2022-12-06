from __future__ import print_function, division

import Functions.Pipeline as pipe


def main(*pipeparts):
    if 'A' in pipeparts:
        pipe.A_Folderize()
    if 'B' in pipeparts:
        train_loader, test_loader = pipe.B_PrepareData()
    if 'C' in pipeparts:
        model = pipe.C_TrainModel(train_loader, test_loader)


if __name__ == '__main__':
    # Define which parts of the pipeline to execute (include 'A' in first execution, then 'A' does not need to be run)
    pipeparts = ['A', 'B', 'C']
    main(*pipeparts)
