task_CFP = {
    "class_incremental":{
        0: list(range(4)),
        1: [4,5],
        2: [6],
    },
    "tasks_name":
    {
        0:"mnms",
        1:"Fundus",
        2:"Prostate",
    }
}

task_PFC = {
    "class_incremental":{
        0: list(range(2)),
        1: [2,3],
        2: [4,5,6],
    },
    "tasks_name":
    {
        0:"Prostate",
        1:"Fundus",
        2:"mnms",
    }
}


def get_task_labels(dataset, mode):
    
    if dataset == 'CFP':
        task_dict = task_CFP[mode]
    if dataset == 'PFC':
        task_dict = task_PFC[mode]
    
    return task_dict

