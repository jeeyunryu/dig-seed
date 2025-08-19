import copy
import importlib

def transform(data, ops=None):
    """transform."""
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data

def create_operators(op_param_list, global_config=None):
    ops = []
    for op_info in op_param_list:
        op_name = list(op_info.keys())[0]
        param = copy.deepcopy(op_info[op_name]) or {}

        if global_config:
            param.update(global_config)

        if op_name in globals():
            op_class = globals()[op_name]
        else:
            op_class = dynamic_import(op_name)

        ops.append(op_class(**param))
    return ops

MODULE_MAPPING = {
    'ABINetLabelEncode': '.abinet_label_encode',
    'ARLabelEncode': '.ar_label_encode',
    'CELabelEncode': '.ce_label_encode',
    'CharLabelEncode': '.char_label_encode',
    'CPPDLabelEncode': '.cppd_label_encode',
    'CTCLabelEncode': '.ctc_label_encode',
    'EPLabelEncode': '.ep_label_encode',
    'IGTRLabelEncode': '.igtr_label_encode',
    'MGPLabelEncode': '.mgp_label_encode',
    'SMTRLabelEncode': '.smtr_label_encode',
    'SRNLabelEncode': '.srn_label_encode',
    'VisionLANLabelEncode': '.visionlan_label_encode',
    'CAMLabelEncode': '.cam_label_encode',
    'ABINetAug': '.rec_aug',
    'BDA': '.rec_aug',
    'PARSeqAug': '.rec_aug',
    'PARSeqAugPIL': '.rec_aug',
    'SVTRAug': '.rec_aug',
    'ABINetResize': '.resize',
    'CDistNetResize': '.resize',
    'LongResize': '.resize',
    'RecTVResize': '.resize',
    'RobustScannerRecResizeImg': '.resize',
    'SliceResize': '.resize',
    'SliceTVResize': '.resize',
    'SRNRecResizeImg': '.resize',
    'SVTRResize': '.resize',
    'VisionLANResize': '.resize',
    'RecDynamicResize': '.resize',
}


def dynamic_import(class_name):
    module_path = MODULE_MAPPING.get(class_name)
    if not module_path:
        raise ValueError(f'Unsupported class: {class_name}')

    module = importlib.import_module(module_path, package=__package__)
    return getattr(module, class_name)