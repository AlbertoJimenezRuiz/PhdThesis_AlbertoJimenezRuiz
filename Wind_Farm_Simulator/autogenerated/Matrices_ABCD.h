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
number_states=11;
number_inputs=4;
number_outputs=5;
const auto x0 = (Llr + Lm);
const auto x1 = (-x0);
const auto x2 = (( (Lm) * (Lm) ));
const auto x3 = (Lls + Lm);
const auto x4 = (-x0*x3 + x2);
const auto x5 = (1.0/x4);
const auto x6 = (x1*x5);
const auto x7 = (Rs*Base_Electrical_angular_speed);
const auto x8 = (-x6*x7);
const auto x9 = (Omega_s*Base_Electrical_angular_speed);
const auto x10 = (Lm*x5);
const auto x11 = (-x10*x7);
const auto x12 = (v_sq*x2);
const auto x13 = (Kp_control*x12);
const auto x14 = (Omega_s*x3);
const auto x15 = (x13*x14);
const auto x16 = (Rr*x12);
const auto x17 = (Base_Electrical_angular_speed*x5);
const auto x18 = (1.0/Lm);
const auto x19 = (1.0/v_sq);
const auto x20 = (x18*x19);
const auto x21 = (x17*x20);
const auto x22 = (1.0/x3);
const auto x23 = (1.0/Omega_s);
const auto x24 = (x21*x22*x23);
const auto x25 = (omega_r - Omega_s);
const auto x26 = (Base_Electrical_angular_speed*x25);
const auto x27 = (Kp_control*x3);
const auto x28 = (-x3);
const auto x29 = (Lm*v_sq);
const auto x30 = (x28*x29);
const auto x31 = (x27*x30);
const auto x32 = (Omega_s*x31);
const auto x33 = (Rr*x30);
const auto x34 = (x14*x4);
const auto x35 = (x29*x34);
const auto x36 = (Omega_s*x4);
const auto x37 = (x25*x36);
const auto x38 = (x30*x37);
const auto x39 = (phi_rq*x3);
const auto x40 = (x29*x39);
const auto x41 = (Lm*phi_sq);
const auto x42 = (-x39 + x41);
const auto x43 = (Lm*x42);
const auto x44 = (v_sq*x43);
const auto x45 = (x36*x44);
const auto x46 = (Ki_control*Base_Electrical_angular_speed);
const auto x47 = (Kp_control*x30);
const auto x48 = (Lm*phi_rd);
const auto x49 = (x4*x48);
const auto x50 = (v_sq*x49);
const auto x51 = (Kopt*x4);
const auto x52 = (2*omega_r);
const auto x53 = (x51*x52);
const auto x54 = (1.0/Hr);
const auto x55 = ((1.0/2.0)*x54);
const auto x56 = (x10*x55);
const auto x57 = ((1.0/2.0)*K_2mass);
const auto x58 = (omega_t <= 0);
const auto x59 = (( (omega_t) * (omega_t) ));
const auto x60 = (K_2mass*Base_Power*theta_shaft);
const auto x61 = (Multiplier*Number_poles);
const auto x62 = (wind_speed*x61);
const auto x63 = (1.0/Base_Electrical_angular_speed);
const auto x64 = (21*x63);
const auto x65 = (x64/omega_t);
const auto x66 = (exp(-x62*x65/Blade_length));
const auto x67 = (M_PI*( (wind_speed) * (wind_speed) * (wind_speed) )*x66);
const auto x68 = (omega_t*Base_Electrical_angular_speed);
const auto x69 = (-4.7236167130238886*Blade_length*x68 + 60.478977782645813*x62);
const auto x70 = (Blade_length*x69);
const auto x71 = (1.0/Ht);
const auto x72 = (x63*x71/Base_Power);
const auto x73 = (1.0/x59);
const auto x74 = (M_PI*x66);
const auto x75 = ((1.0/2.0)*x72*x73);
const auto x76 = (-x10);
const auto x77 = (-x28*x5);
const auto x78 = (x20*x3);
const auto x79 = (Ycr*x13);
const auto x80 = (( (v_sq) * (v_sq) ));
const auto x81 = (-x4);
const auto x82 = (Lm*phi_sd - phi_rd*x3);
const auto x83 = (Omega_s*x82);
const auto x84 = (x29*x83);
const auto x85 = (desired_reactive*x34 + x84);
const auto x86 = (state_control_reactive*Ki_control);
const auto x87 = (x25*x45 - x35*x86);
const auto x88 = (x27*(x80*x81 + x85) + x87);
const auto x89 = (x3*x84);
const auto x90 = (Ycr*x88);
const auto x91 = (x89 + x90);
const auto x92 = (-x15*x91 - x88*(x12*x14 + x14*x79));
const auto x93 = (1.0/x80);
const auto x94 = (( (v_sd) * (v_sd) ) + x80);
const auto x95 = (1.0/x94);
const auto x96 = (v_sd*x95);
const auto x97 = (( (x4) * (x4) ));
const auto x98 = (1.0/x97);
const auto x99 = (1.0/x2);
const auto x100 = (( (Omega_s) * (Omega_s) ));
const auto x101 = (1.0/x100);
const auto x102 = (( (x3) * (x3) ));
const auto x103 = (1.0/x102);
const auto x104 = (x101*x103*x98*x99);
const auto x105 = (x104*x93*x96);
const auto x106 = (state_control_active*Ki_control);
const auto x107 = (x29*x4);
const auto x108 = (( (omega_r) * (omega_r) )*x3*x51 - x44);
const auto x109 = (Kp_control*x108 + x106*x107);
const auto x110 = (Ycr*x109);
const auto x111 = (-x110 + x44);
const auto x112 = (x100*x102);
const auto x113 = (x111*x112);
const auto x114 = (x12*x37);
const auto x115 = (x100*x102*x109*(x12 + x79) - x113*x13 - x114*x90 - x114*x91);
const auto x116 = (-x32*x91 - x88*(Ycr*x32 + x14*x30));
const auto x117 = (x100*x102*x109*(Ycr*x47 + x30) - x113*x47 - x38*x90 - x38*x91);
const auto x118 = (( (x3) * (x3) * (x3) ));
const auto x119 = (2*Kopt*Kp_control*omega_r*x100*x111*x118*x4 - Kp_control*x100*x110*x118*x53 - x45*x90 - x45*x91);
const auto x120 = (Ki_control*x35);
const auto x121 = (x120*x90 + x120*x91);
const auto x122 = (Ki_control*Lm*v_sq*x100*x102*x111*x4 - Ki_control*x107*x110*x112);
const auto x123 = (-GSC_pole_filter);
const auto x124 = (x104*x95);
const auto x125 = (x124*x19);
const auto x126 = (-x17*x81);
const auto x127 = (Lm*x83);
const auto x128 = (2*v_sq);
const auto x129 = (x127 - x128*x4);
const auto x130 = (x127*x3);
const auto x131 = (Lm*x4);
const auto x132 = (Lm*x34);
const auto x133 = (-x132*x86 + x37*x43);
const auto x134 = (-x4*x80 + x85);
const auto x135 = (x20*x27);
const auto x136 = (-Kp_control*x43 + x106*x131);
const auto x137 = (x18*x93);
const auto x138 = (x61*x67);
const auto x139 = (x20*x23);
const auto x140 = (x137*x5);
const auto x141 = (x42*x5);
const auto x142 = (GSC_pole_filter*rsc_current_d);
const auto x143 = (x142*x80);
const auto x144 = (x112*x2*x97);
const auto x145 = (2*x144);
const auto x146 = (x88*x91);
const auto x147 = (-x100*x102*x109*x111 + x146);
const auto x148 = (x144*x94);
const auto x149 = (x109*x112);
const auto x150 = (v_sd*(x111*x149 - x146) - x143*x148);
const auto x151 = (2*x104/( (x94) * (x94) ));
const auto x152 = (x150*x151);
const auto x153 = (( (v_sq) * (v_sq) * (v_sq) ));
const auto x154 = (x112*x12*x94*x97);
const auto x155 = (-Ycr*x136 + x43);
const auto x156 = (x133 + x27*(x127 + x128*x81));
const auto x157 = (x88*(Ycr*x156 + x130));
const auto x158 = (x156*x91);
const auto x159 = (Kp_control*x102);
const auto x160 = (x159*x36);
const auto x161 = (-x160*x90 - x160*x91);
const auto x162 = (GSC_pole_filter*rsc_current_q);
const auto x163 = (2*x162);
const auto x164 = (-x147 - x154*x162);
const auto x165 = (x151*x164);
const auto x166 = (x22*x98);
const auto x167 = (x139*x166);
const auto x168 = (x14*x42);
const auto x169 = (x4*x83);
const auto x170 = (x169*x25);
const auto x171 = (x5*x82);
const auto x172 = (-phi_sd*x0 + x48);
const auto x173 = (x172*x5);
const auto x174 = (Lm*phi_rq - phi_sq*x0);
const auto x175 = (v_sq*x174);
const auto x176 = (v_sd*x172 + x175);
MatA[0] = (x8);
MatA[1] = (x9);
MatA[2] = (x11);
MatA[3] = (0);
MatA[4] = (0);
MatA[5] = (0);
MatA[6] = (0);
MatA[7] = (0);
MatA[8] = (0);
MatA[9] = (0);
MatA[10] = (0);
MatA[11] = (-x9);
MatA[12] = (x8);
MatA[13] = (0);
MatA[14] = (x11);
MatA[15] = (0);
MatA[16] = (0);
MatA[17] = (0);
MatA[18] = (0);
MatA[19] = (0);
MatA[20] = (0);
MatA[21] = (0);
MatA[22] = (-x24*(x14*x16 + x15));
MatA[23] = (-Lm*x22*x26);
MatA[24] = (-x24*(x14*x33 + x32));
MatA[25] = (-x24*(x25*x35 + x38));
MatA[26] = (-x24*(x36*x40 + x45));
MatA[27] = (0);
MatA[28] = (0);
MatA[29] = (x46);
MatA[30] = (0);
MatA[31] = (0);
MatA[32] = (0);
MatA[33] = (0);
MatA[34] = (x21*(-x13 - x16));
MatA[35] = (x26);
MatA[36] = (x21*(-x33 - x47));
MatA[37] = (x21*(x27*x53 + x50));
MatA[38] = (0);
MatA[39] = (0);
MatA[40] = (0);
MatA[41] = (x46);
MatA[42] = (0);
MatA[43] = (0);
MatA[44] = (phi_rq*x56);
MatA[45] = (-phi_rd*x56);
MatA[46] = (-x41*x5*x55);
MatA[47] = (phi_sd*x56);
MatA[48] = (0);
MatA[49] = (0);
MatA[50] = (-x54*x57);
MatA[51] = (0);
MatA[52] = (0);
MatA[53] = (0);
MatA[54] = (0);
MatA[55] = (0);
MatA[56] = (0);
MatA[57] = (0);
MatA[58] = (0);
MatA[59] = (0);
MatA[60] = (((x58) ? (
   0
)
: (
   x75*(-4.7236167130238886*( (Blade_length) * (Blade_length) )*Base_Electrical_angular_speed*x67 + ( (wind_speed) * (wind_speed) * (wind_speed) * (wind_speed) )*x61*x64*x69*x73*x74 + 2*x60*x68) - x72*(Base_Electrical_angular_speed*x59*x60 + x67*x70)/( (omega_t) * (omega_t) * (omega_t) )
)));
MatA[61] = (x57*x71);
MatA[62] = (0);
MatA[63] = (0);
MatA[64] = (0);
MatA[65] = (0);
MatA[66] = (0);
MatA[67] = (0);
MatA[68] = (0);
MatA[69] = (0);
MatA[70] = (Base_Electrical_angular_speed);
MatA[71] = (-Base_Electrical_angular_speed);
MatA[72] = (0);
MatA[73] = (0);
MatA[74] = (0);
MatA[75] = (0);
MatA[76] = (0);
MatA[77] = (x76);
MatA[78] = (0);
MatA[79] = (x77);
MatA[80] = (0);
MatA[81] = (0);
MatA[82] = (0);
MatA[83] = (0);
MatA[84] = (0);
MatA[85] = (0);
MatA[86] = (0);
MatA[87] = (0);
MatA[88] = (0);
MatA[89] = (x76);
MatA[90] = (0);
MatA[91] = (x77);
MatA[92] = (Kopt*x52*x78);
MatA[93] = (0);
MatA[94] = (0);
MatA[95] = (0);
MatA[96] = (0);
MatA[97] = (0);
MatA[98] = (0);
MatA[99] = (x105*x92);
MatA[100] = (x105*x115);
MatA[101] = (x105*x116);
MatA[102] = (x105*x117);
MatA[103] = (x105*x119);
MatA[104] = (0);
MatA[105] = (0);
MatA[106] = (x105*x121);
MatA[107] = (x105*x122);
MatA[108] = (x123);
MatA[109] = (0);
MatA[110] = (x125*x92);
MatA[111] = (x115*x125);
MatA[112] = (x116*x125);
MatA[113] = (x117*x125);
MatA[114] = (x119*x125);
MatA[115] = (0);
MatA[116] = (0);
MatA[117] = (x121*x125);
MatA[118] = (x122*x125);
MatA[119] = (0);
MatA[120] = (x123);
MatB[0] = (x126);
MatB[1] = (0);
MatB[2] = (0);
MatB[3] = (0);
MatB[4] = (0);
MatB[5] = (x126);
MatB[6] = (0);
MatB[7] = (0);
MatB[8] = (0);
MatB[9] = (Base_Electrical_angular_speed*x18*x22*x23*x5*x93*(Rr*x89 + x134*x27 + x37*x40 + x87) - x24*(Rr*x130 + Omega_s*x131*x25*x39 + x129*x27 + x133));
MatB[10] = (-Base_Electrical_angular_speed*x135);
MatB[11] = (0);
MatB[12] = (0);
MatB[13] = (-x137*x17*(-Rr*x44 + x109 + x25*x50) + x21*(-Rr*x43 + x136 + x25*x49));
MatB[14] = (0);
MatB[15] = (0);
MatB[16] = (0);
MatB[17] = (0);
MatB[18] = (0);
MatB[19] = (0);
MatB[20] = (0);
MatB[21] = (0);
MatB[22] = (0);
MatB[23] = (((x58) ? (
   0
)
: (
   x75*(60.478977782645813*Blade_length*x138 + 3*( (wind_speed) * (wind_speed) )*x70*x74 - x138*x65*x69)
)));
MatB[24] = (0);
MatB[25] = (0);
MatB[26] = (0);
MatB[27] = (0);
MatB[28] = (0);
MatB[29] = (-x129*x139*x5 + x134*x140*x23);
MatB[30] = (-x78);
MatB[31] = (0);
MatB[32] = (0);
MatB[33] = (-x108*x140 - x141*x19);
MatB[34] = (0);
MatB[35] = (0);
MatB[36] = (-v_sd*x152*x93 + x101*x103*x93*x95*x98*x99*(-v_sd*x143*x145 - x147));
MatB[37] = (x101*x103*x93*x95*x98*x99*(v_sd*(x113*x136 + x149*x155 - x157 - x158) - x142*x145*x153 - 2*x142*x154) - 2*x124*x150/x153 - x152*x19);
MatB[38] = (x105*x161);
MatB[39] = (0);
MatB[40] = (-v_sd*x165*x19 - x163*x96);
MatB[41] = (x101*x103*x19*x95*x98*x99*(x100*x102*x109*x155 + x100*x102*x111*x136 - x144*x163*x80 - x148*x162 - x157 - x158) - x124*x164*x93 - x165);
MatB[42] = (x125*x161);
MatB[43] = (0);
MatC[0] = (x6);
MatC[1] = (0);
MatC[2] = (x10);
MatC[3] = (0);
MatC[4] = (0);
MatC[5] = (0);
MatC[6] = (0);
MatC[7] = (0);
MatC[8] = (0);
MatC[9] = (GSC_pole_filter);
MatC[10] = (0);
MatC[11] = (0);
MatC[12] = (x6);
MatC[13] = (0);
MatC[14] = (x10);
MatC[15] = (0);
MatC[16] = (0);
MatC[17] = (0);
MatC[18] = (0);
MatC[19] = (0);
MatC[20] = (0);
MatC[21] = (GSC_pole_filter);
MatC[22] = (0);
MatC[23] = (0);
MatC[24] = (0);
MatC[25] = (0);
MatC[26] = (1);
MatC[27] = (0);
MatC[28] = (0);
MatC[29] = (0);
MatC[30] = (0);
MatC[31] = (0);
MatC[32] = (0);
MatC[33] = (x167*(Lm*Omega_s*v_sd*v_sq*x1*x3*x4 - Lm*x88 - x13*x3*x83));
MatC[34] = (x167*(Lm*Omega_s*x1*x3*x4*x80 + Lm*Omega_s*x109*x3 - x12*x170 - x13*x168));
MatC[35] = (x167*(Omega_s*v_sd*v_sq*x2*x3*x4 - x28*x88 - x31*x83));
MatC[36] = (x167*(Omega_s*x109*x28*x3 + Omega_s*x2*x3*x4*x80 - Omega_s*x27*x28*x44 - x170*x30));
MatC[37] = (x167*(Omega_s*x159*x42*x53 - x169*x44));
MatC[38] = (0);
MatC[39] = (0);
MatC[40] = (Ki_control*x171);
MatC[41] = (Ki_control*x141);
MatC[42] = (0);
MatC[43] = (0);
MatC[44] = (v_sq*x6);
MatC[45] = (-v_sd*x6);
MatC[46] = (v_sq*x10);
MatC[47] = (-v_sd*x10);
MatC[48] = (0);
MatC[49] = (0);
MatC[50] = (0);
MatC[51] = (0);
MatC[52] = (0);
MatC[53] = (0);
MatC[54] = (0);
MatD[0] = (0);
MatD[1] = (0);
MatD[2] = (0);
MatD[3] = (0);
MatD[4] = (0);
MatD[5] = (0);
MatD[6] = (0);
MatD[7] = (0);
MatD[8] = (0);
MatD[9] = (0);
MatD[10] = (0);
MatD[11] = (0);
MatD[12] = (x173);
MatD[13] = (-x137*x166*x23*(x109*x14*x42 + x176*x35 - x82*x88) + x167*(x132*x175 + x132*x176 + x136*x168 - x156*x82));
MatD[14] = (-x135*x171);
MatD[15] = (0);
MatD[16] = (-x174*x5);
MatD[17] = (x173);
MatD[18] = (0);
MatD[19] = (0);
