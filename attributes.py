class Attributes:

    T_attribute = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]

    Rh_attribute = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9"]

    condition_attr = ["T_out", "Tdewpoint", "RH_out", "Press_mm_hg",
                   "Windspeed", "Visibility"]
    light_attribute = ["lights"]

    rv_attribute = ["rv1", "rv2"]

    predictor_attribute = ["Appliances"]

    T_ONLY = predictor_attribute + T_attribute

    RH_ONLY = predictor_attribute +Rh_attribute

    COND_ONLY = predictor_attribute + condition_attr

    RV_ONLY = predictor_attribute + rv_attribute

    ALL_FEATURES = predictor_attribute + T_attribute + Rh_attribute + condition_attr + rv_attribute

    BEST_FIT = predictor_attribute + T_attribute + Rh_attribute