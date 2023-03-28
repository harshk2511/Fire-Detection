def calcfinalconf2(output):
    fire_conf = output[0][0].item()
    non_fire_conf = output[0][1].item()
    if (fire_conf >= non_fire_conf):
        print("FIRE")
        final_cat = "Fire"
    else:
        print("NON-FIRE")
        final_cat = "Non-Fire"
    
    return final_cat

def calcfinalconf1(output):
    fire_conf = abs(output[0][0].item())
    non_fire_conf = abs(output[0][1].item())
    print(fire_conf)
    print(non_fire_conf)


    diff_f_nf = abs(fire_conf - non_fire_conf)
    final_conf_val = max(fire_conf, non_fire_conf)
    if (diff_f_nf >= 0.049):
        if (final_conf_val == fire_conf):
            final_cat = "Non-Fire"
            print("2. FINAL CATEGORY: NON_FIRE")
        else:
            final_cat = "Fire"
            print("2. FINAL CATEGORY: FIRE")
    else:
    # final_conf_val = max(fire_conf, non_fire_conf) 
        if (final_conf_val == fire_conf):
            final_cat = "Fire"
            print("FINAL CATEGORY: FIRE")
        else:
            final_cat = "Non-Fire"
            print("FINAL CATEGORY: NON_FIRE")
    
    return final_cat

