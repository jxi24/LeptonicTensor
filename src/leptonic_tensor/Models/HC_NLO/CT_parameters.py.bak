# This file was automatically created by FeynRules $Revision: 535 $
# Mathematica version: 7.0 for Mac OS X x86 (64-bit) (November 11, 2008)
# Date: Fri 18 Mar 2011 18:40:51

# Modified by F. Demartin in order to include loop Higgs EFT
# Dec 2013



from object_library import all_CTparameters, CTParameter

from function_library import complexconjugate, re, im, csc, sec, acsc, asec

################
# R2 vertices  #
################

# ========= #
# Pure QCD  #
# ========= #

RGR2 = CTParameter(name = 'RGR2',
              type = 'real',
              value = {0:'-(3.0/2.0)*G**4/(96.0*cmath.pi**2)'},
              texname = 'RGR2')

# ============== #
# Mixed QCD-QED  #
# ============== #

R2MixedFactor = CTParameter(name = 'R2MixedFactor',
              type = 'real',
              value = {0:'-(G**2*(1.0+lhv)*(Ncol**2-1.0))/(2.0*Ncol*16.0*cmath.pi**2)'},
              texname = 'R2MixedFactor')

################
# UV vertices  #
################

# ========= #
# Pure QCD  #
# ========= #

# gs coupling renormalisation parameters

G_UVg = CTParameter(name = 'G_UVg',
                    type = 'real',
                    value = {-1:'-((G**2)/(2.0*48.0*cmath.pi**2))*11.0*CA'},
                    texname = '\delta Gg')

G_UVq = CTParameter(name = 'G_UVq',
                    type = 'real',
                    value = {-1:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF'},
                    texname = '\delta Gq')

G_UVc = CTParameter(name = 'G_UVc',
                    type = 'real',
                    value = {-1:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF',
                              0:'cond(MC,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MC**2/MU_R**2))'},
                    texname = '\delta Gc')

G_UVb = CTParameter(name = 'G_UVb',
                    type = 'real',
                    value = {-1:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF',
                              0:'cond(MB,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MB**2/MU_R**2))'},
                    texname = '\delta Gb')

G_UVt = CTParameter(name = 'G_UVt',
                    type = 'real',
                    value = {-1:'((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF',
                              0:'cond(MT,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MT**2/MU_R**2))'},
                    texname = '\delta Gt')


# gluon wavefunction renormalisation parameters

GWcft_UV_c = CTParameter(name = 'GWcft_UV_c',
                         type = 'real',
                         value = {-1:'cond(MC,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF)',
                                   0:'cond(MC,0.0,((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MC**2/MU_R**2))'
                                 },
                         texname = '\delta G_{wfct\_c}')

GWcft_UV_b = CTParameter(name = 'GWcft_UV_b',
                         type = 'real',
                         value = {-1:'cond(MB,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF)',
                                   0:'cond(MB,0.0,((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MB**2/MU_R**2))'
                                 },
                         texname = '\delta G_{wfct\_b}')

GWcft_UV_t = CTParameter(name = 'GWcft_UV_t',
                         type = 'real',
                         value = {-1:'cond(MT,0.0,-((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF)',
                                   0:'cond(MT,0.0,((G**2)/(2.0*48.0*cmath.pi**2))*4.0*TF*reglog(MT**2/MU_R**2))' },
                         texname = '\delta G_{wfct\_t}')


# massive quark wavefunction renormalisation parameters

cWcft_UV = CTParameter(name = 'cWcft_UV',
                       type = 'real',
                       value = {-1:'cond(MC,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*3.0*CF)',
                                 0:'cond(MC,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MC**2/MU_R**2)))'
                               },
                       texname = '\delta Z_c')

bWcft_UV = CTParameter(name = 'bWcft_UV',
                       type = 'real',
                       value = {-1:'cond(MB,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*3.0*CF)',
                                 0:'cond(MB,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MB**2/MU_R**2)))'
                               },
                       texname = '\delta Z_b')

tWcft_UV = CTParameter(name = 'tWcft_UV',
                       type = 'real',
                       value = {-1:'cond(MT,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*3.0*CF)',
                                 0:'cond(MT,0.0,-((G**2)/(2.0*16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MT**2/MU_R**2)))' },
                       texname = '\delta Z_t')


# quark mass renormalisation parameters

bMass_UV = CTParameter(name = 'bMass_UV',
                       type = 'complex',
                       value = {-1:'cond(MB,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*(3.0*CF)*MB)',
                                 0:'cond(MB,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MB**2/MU_R**2))*MB)'
                               },
                       texname = '\delta m_b')

cMass_UV = CTParameter(name = 'cMass_UV',
                       type = 'complex',
                       value = {-1:'cond(MC,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*(3.0*CF)*MC)',
                                 0:'cond(MC,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MC**2/MU_R**2))*MC)'
                               },
                       texname = '\delta m_c')

tMass_UV = CTParameter(name = 'tMass_UV',
                       type = 'complex',
                       value = {-1:'cond(MT,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*MT)',
                                 0:'cond(MT,0.0,complex(0,1)*((G**2)/(16.0*cmath.pi**2))*CF*(4.0-3.0*reglog(MT**2/MU_R**2))*MT)' },
                       texname = '\delta m_t')

# ============== #
# Mixed QCD-QED  #
# ============== #


# yukawas renormalisation for c, b, t
UV_yuk_c = CTParameter(name = 'UV_yuk_c',
                       type = 'real',
                       value = {-1:'-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*2.0',
                                 0:'cond(MC,0.0,-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*CF*(-3.0*reglog(MC**2/MU_R**2)+4.0)*2.0)' },
                       texname = '\delta y_c')

UV_yuk_b = CTParameter(name = 'UV_yuk_b',
                       type = 'real',
                       value = {-1:'-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*2.0',
                                 0:'cond(MB,0.0,-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*CF*(-3.0*reglog(MB**2/MU_R**2)+4.0)*2.0)' },
                       texname = '\delta y_b')

UV_yuk_t = CTParameter(name = 'UV_yuk_t',
                       type = 'real',
                       value = {-1:'-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*3.0*CF*2.0',
                                 0:'cond(MT,0.0,-(1.0/2.0)*((G**2)/(16.0*cmath.pi**2))*CF*(-3.0*reglog(MT**2/MU_R**2)+4.0)*2.0)' },
                       texname = '\delta y_t')







#************************************************************#
# NEW
# UV renormalisation parameters for X0
#************************************************************#



### ggX0 UV parameters ###


G_UV_ggX0_hq = CTParameter(name = 'G_UV_ggX0_hq',
                           type = 'complex',
                           value = {-1:'-complex(0,1)/2.*cosa*kHgg*gHgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)'},
                           texname = '\delta gsGHq')

G_UV_ggX0_hc = CTParameter(name = 'G_UV_ggX0_hc',
                           type = 'complex',
                           value = {-1:'-complex(0,1)/2.*cosa*kHgg*gHgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MC, 0.0, complex(0,1)/2.*cosa*kHgg*gHgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)*reglog(MC**2/MU_R**2) )'},
                           texname = '\delta gsGHc')

G_UV_ggX0_hb = CTParameter(name = 'G_UV_ggX0_hb',
                           type = 'complex',
                           value = {-1:'-complex(0,1)/2.*cosa*kHgg*gHgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MB, 0.0, complex(0,1)/2.*cosa*kHgg*gHgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)*reglog(MB**2/MU_R**2) )'},
                           texname = '\delta gsGHb')

G_UV_ggX0_ht = CTParameter(name = 'G_UV_ggX0_ht',
                           type = 'complex',
                           value = {-1:'-complex(0,1)/2.*cosa*kHgg*gHgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MT, 0.0, complex(0,1)/2.*cosa*kHgg*gHgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)*reglog(MT**2/MU_R**2) )'},
                           texname = '\delta gsGHt')

G_UV_ggX0_hg = CTParameter(name = 'G_UV_ggX0_hg',
                           type = 'complex',
                           value = {-1:'complex(0,1)/2.*cosa*kHgg*gHgg*G**2/(4.*cmath.pi**2)*(11.*CA/6.)',
                                     0:'-complex(0,1)*11./4.*cosa*kHgg*gHgg*G**2/(4.*cmath.pi**2)'},
                           texname = '\delta gsGHg')



G_UV_ggX0_aq = CTParameter(name = 'G_UV_ggX0_aq',
                           type = 'complex',
                           value = {-1:'-complex(0,1)/2.*sina*kAgg*gAgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)'},
                           texname = '\delta gsGA0q')

G_UV_ggX0_ac = CTParameter(name = 'G_UV_ggX0_ac',
                           type = 'complex',
                           value = {-1:'-complex(0,1)/2.*sina*kAgg*gAgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MC, 0.0, complex(0,1)/2.*sina*kAgg*gAgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)*reglog(MC**2/MU_R**2) )'},
                           texname = '\delta gsGA0c')

G_UV_ggX0_ab = CTParameter(name = 'G_UV_ggX0_ab',
                           type = 'complex',
                           value = {-1:'-complex(0,1)/2.*sina*kAgg*gAgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MB, 0.0, complex(0,1)/2.*sina*kAgg*gAgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)*reglog(MB**2/MU_R**2) )'},
                           texname = '\delta gsGA0b')

G_UV_ggX0_at = CTParameter(name = 'G_UV_ggX0_at',
                           type = 'complex',
                           value = {-1:'-complex(0,1)/2.*sina*kAgg*gAgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MT, 0.0, complex(0,1)/2.*sina*kAgg*gAgg*G**2/(4.*cmath.pi**2)*(2.*TF/3.)*reglog(MT**2/MU_R**2) )'},
                           texname = '\delta gsGA0t')

G_UV_ggX0_ag = CTParameter(name = 'G_UV_ggX0_ag',
                           type = 'complex',
                           value = {-1:'complex(0,1)/2.*sina*kAgg*gAgg*G**2/(4.*cmath.pi**2)*(11.*CA/6.)'},
                           texname = '\delta gsGA0g')







### gggX0 UV parameters ###


G_UV_3gX0_hq = CTParameter(name = 'G_UV_3gX0_hq',
                           type = 'complex',
                           value = {-1:'-(3.*cosa*kHgg*gHgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)'},
                           texname = '\delta gsGHq')

G_UV_3gX0_hc = CTParameter(name = 'G_UV_3gX0_hc',
                           type = 'complex',
                           value = {-1:'-(3.*cosa*kHgg*gHgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MC, 0.0, (3.*cosa*kHgg*gHgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)*reglog(MC**2/MU_R**2) )'},
                           texname = '\delta gsGHc')

G_UV_3gX0_hb = CTParameter(name = 'G_UV_3gX0_hb',
                           type = 'complex',
                           value = {-1:'-(3.*cosa*kHgg*gHgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MB, 0.0, (3.*cosa*kHgg*gHgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)*reglog(MB**2/MU_R**2) )'},
                           texname = '\delta gsGHb')

G_UV_3gX0_ht = CTParameter(name = 'G_UV_3gX0_ht',
                           type = 'complex',
                           value = {-1:'-(3.*cosa*kHgg*gHgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MT, 0.0, (3.*cosa*kHgg*gHgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)*reglog(MT**2/MU_R**2) )'},
                           texname = '\delta gsGHt')

G_UV_3gX0_hg = CTParameter(name = 'G_UV_3gX0_hg',
                           type = 'complex',
                           value = {-1:'(3.*cosa*kHgg*gHgg*G**3)/(16.*cmath.pi**2)*(11.*CA/6.)',
                                     0:'(-11.*cosa*kHgg*gHgg*G**3)/(16.*cmath.pi**2)'},
                           texname = '\delta gsGHg')



G_UV_3gX0_aq = CTParameter(name = 'G_UV_3gX0_aq',
                           type = 'complex',
                           value = {-1:'-(3.*sina*kAgg*gAgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)'},
                           texname = '\delta gsGA0q')

G_UV_3gX0_ac = CTParameter(name = 'G_UV_3gX0_ac',
                           type = 'complex',
                           value = {-1:'-(3.*sina*kAgg*gAgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MC, 0.0, (3.*sina*kAgg*gAgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)*reglog(MC**2/MU_R**2) )'},
                           texname = '\delta gsGA0c')

G_UV_3gX0_ab = CTParameter(name = 'G_UV_3gX0_ab',
                           type = 'complex',
                           value = {-1:'-(3.*sina*kAgg*gAgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MB, 0.0, (3.*sina*kAgg*gAgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)*reglog(MB**2/MU_R**2) )'},
                           texname = '\delta gsGA0b')

G_UV_3gX0_at = CTParameter(name = 'G_UV_3gX0_at',
                           type = 'complex',
                           value = {-1:'-(3.*sina*kAgg*gAgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MT, 0.0, (3.*sina*kAgg*gAgg*G**3)/(16.*cmath.pi**2)*(2.*TF/3.)*reglog(MT**2/MU_R**2) )'},
                           texname = '\delta gsGA0t')

G_UV_3gX0_ag = CTParameter(name = 'G_UV_3gX0_ag',
                           type = 'complex',
                           value = {-1:'(3.*sina*kAgg*gAgg*G**3)/(16.*cmath.pi**2)*(11.*CA/6.)'},
                           texname = '\delta gsGA0g')







### ggggX0 UV parameters ###


G_UV_4gX0_hq = CTParameter(name = 'G_UV_4gX0_hq',
                           type = 'complex',
                           value = {-1:'complex(0,1)*(cosa*kHgg*gHgg*G**4)/(4.*cmath.pi**2)*(2.*TF/3.)'},
                           texname = '\delta gs^2GHq')

G_UV_4gX0_hc = CTParameter(name = 'G_UV_4gX0_hc',
                           type = 'complex',
                           value = {-1:'complex(0,1)*(cosa*kHgg*gHgg*G**4)/(4.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MC, 0.0, -complex(0,1)*(cosa*kHgg*gHgg*G**4)/(4.*cmath.pi**2)*(2.*TF/3.)*reglog(MC**2/MU_R**2) )'},
                           texname = '\delta gs^2GHc')

G_UV_4gX0_hb = CTParameter(name = 'G_UV_4gX0_hb',
                           type = 'complex',
                           value = {-1:'complex(0,1)*(cosa*kHgg*gHgg*G**4)/(4.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MB, 0.0, -complex(0,1)*(cosa*kHgg*gHgg*G**4)/(4.*cmath.pi**2)*(2.*TF/3.)*reglog(MB**2/MU_R**2) )'},
                           texname = '\delta gs^2GHb')

G_UV_4gX0_ht = CTParameter(name = 'G_UV_4gX0_ht',
                           type = 'complex',
                           value = {-1:'complex(0,1)*(cosa*kHgg*gHgg*G**4)/(4.*cmath.pi**2)*(2.*TF/3.)',
                                     0:'cond( MT, 0.0, -complex(0,1)*(cosa*kHgg*gHgg*G**4)/(4.*cmath.pi**2)*(2.*TF/3.)*reglog(MT**2/MU_R**2) )'},
                           texname = '\delta gs^2GHt')

G_UV_4gX0_hg = CTParameter(name = 'G_UV_4gX0_hg',
                           type = 'complex',
                           value = {-1:'-complex(0,1)*(cosa*kHgg*gHgg*G**4)/(4.*cmath.pi**2)*(11.*CA/6.)',
                                     0:'complex(0,1)*(11*cosa*kHgg*gHgg*G**4)/(16.*cmath.pi**2)'},
                           texname = '\delta gs^2GHg')







### qq~X0 UV parameters ###

### finite part????
### corresponding for A0????

G_UV_cxcX0_h_mass = CTParameter(name = 'G_UV_cxcX0_h_mass',
                                type = 'complex',
                                value = {-1:'-complex(0,1)*(3.*cosa*kHgg*gHgg*G**2)/(16.*cmath.pi**2)*(CF*MC)'},
                                texname = '\delta ycHEFT')

G_UV_bxbX0_h_mass = CTParameter(name = 'G_UV_bxbX0_h_mass',
                                type = 'complex',
                                value = {-1:'-complex(0,1)*(3.*cosa*kHgg*gHgg*G**2)/(16.*cmath.pi**2)*(CF*MB)'},
                                texname = '\delta ybHEFT')

G_UV_txtX0_h_mass = CTParameter(name = 'G_UV_txtX0_h_mass',
                                type = 'complex',
                                value = {-1:'-complex(0,1)*(3.*cosa*kHgg*gHgg*G**2)/(16.*cmath.pi**2)*(CF*MT)'},
                                texname = '\delta ytHEFT')



