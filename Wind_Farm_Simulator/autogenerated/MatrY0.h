const decltype(wt.Lm) Lm=wt.Lm; (void)Lm;
const decltype(wt.Llr) Llr=wt.Llr; (void)Llr;
const decltype(wt.Lls) Lls=wt.Lls; (void)Lls;
const decltype(wt.Rs) Rs=wt.Rs; (void)Rs;
const decltype(wt.Rr) Rr=wt.Rr; (void)Rr;
const decltype(wt.Ycr) Ycr=wt.Ycr; (void)Ycr;
const decltype(wt.Friction_factor) Friction_factor=wt.Friction_factor; (void)Friction_factor;
const decltype(wt.Base_Electrical_angular_speed) Base_Electrical_angular_speed=wt.Base_Electrical_angular_speed; (void)Base_Electrical_angular_speed;
const decltype(wt.Omega_s) Omega_s=wt.Omega_s; (void)Omega_s;
const decltype(wt.Number_poles) Number_poles=wt.Number_poles; (void)Number_poles;
const decltype(wt.Kp_control) Kp_control=wt.Kp_control; (void)Kp_control;
const decltype(wt.Ki_control) Ki_control=wt.Ki_control; (void)Ki_control;
const decltype(wt.Hr) Hr=wt.Hr; (void)Hr;
const decltype(wt.Ht) Ht=wt.Ht; (void)Ht;
const decltype(wt.K_2mass) K_2mass=wt.K_2mass; (void)K_2mass;
const decltype(wt.Kopt) Kopt=wt.Kopt; (void)Kopt;
const decltype(wt.Blade_length) Blade_length=wt.Blade_length; (void)Blade_length;
const decltype(wt.Multiplier) Multiplier=wt.Multiplier; (void)Multiplier;
const decltype(wt.Base_Power) Base_Power=wt.Base_Power; (void)Base_Power;
const decltype(wt.GSC_pole_filter) GSC_pole_filter=wt.GSC_pole_filter; (void)GSC_pole_filter;
const decltype(wt.Nominal_mechanical_speed_RPM) Nominal_mechanical_speed_RPM=wt.Nominal_mechanical_speed_RPM; (void)Nominal_mechanical_speed_RPM;
const decltype(wt.Base_Voltage) Base_Voltage=wt.Base_Voltage; (void)Base_Voltage;
const decltype(input[0]) v_sd=input[0]; (void)v_sd;
const decltype(input[1]) v_sq=input[1]; (void)v_sq;
const decltype(input[2]) desired_reactive=input[2]; (void)desired_reactive;
const decltype(input[3]) wind_speed=input[3]; (void)wind_speed;
const decltype(state[0]) phi_sd=state[0]; (void)phi_sd;
const decltype(state[1]) phi_sq=state[1]; (void)phi_sq;
const decltype(state[2]) phi_rd=state[2]; (void)phi_rd;
const decltype(state[3]) phi_rq=state[3]; (void)phi_rq;
const decltype(state[4]) omega_r=state[4]; (void)omega_r;
const decltype(state[5]) omega_t=state[5]; (void)omega_t;
const decltype(state[6]) theta_shaft=state[6]; (void)theta_shaft;
const decltype(state[7]) state_control_reactive=state[7]; (void)state_control_reactive;
const decltype(state[8]) state_control_active=state[8]; (void)state_control_active;
const decltype(state[9]) rsc_current_d=state[9]; (void)rsc_current_d;
const decltype(state[10]) rsc_current_q=state[10]; (void)rsc_current_q;
output[0] = ((Lm*phi_rd - phi_sd*(Llr + Lm) + GSC_pole_filter*rsc_current_d*(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm)))/(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm)));
output[1] = ((Lm*phi_rq - phi_sq*(Llr + Lm) + GSC_pole_filter*rsc_current_q*(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm)))/(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm)));
output[2] = (omega_r);
output[3] = ((Lm*Omega_s*v_sq*(Lls + Lm)*(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm))*(v_sd*(Lm*phi_rd - phi_sd*(Llr + Lm)) + v_sq*(Lm*phi_rq - phi_sq*(Llr + Lm))) + Omega_s*(Lls + Lm)*(Lm*phi_sq - phi_rq*(Lls + Lm))*(state_control_active*Ki_control*Lm*v_sq*(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm)) + Kp_control*(Kopt*( (omega_r) * (omega_r) )*(Lls + Lm)*(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm)) - Lm*v_sq*(Lm*phi_sq - phi_rq*(Lls + Lm)))) - (Lm*phi_sd - phi_rd*(Lls + Lm))*(-state_control_reactive*Ki_control*Lm*Omega_s*v_sq*(Lls + Lm)*(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm)) + Kp_control*(Lls + Lm)*(Lm*Omega_s*v_sq*(Lm*phi_sd - phi_rd*(Lls + Lm)) + desired_reactive*Omega_s*(Lls + Lm)*(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm)) + ( (v_sq) * (v_sq) )*(-( (Lm) * (Lm) ) + (Llr + Lm)*(Lls + Lm))) + Lm*Omega_s*v_sq*(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm))*(omega_r - Omega_s)*(Lm*phi_sq - phi_rq*(Lls + Lm))))/(Lm*Omega_s*v_sq*(Lls + Lm)*pow(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm), 2)));
output[4] = ((-v_sd*(Lm*phi_rq - phi_sq*(Llr + Lm)) + v_sq*(Lm*phi_rd - phi_sd*(Llr + Lm)))/(( (Lm) * (Lm) ) - (Llr + Lm)*(Lls + Lm)));