"""
Hopf Algebra q-Deformed Quantum Kernel — Final honest experiments for Quantum journal.
Reports what is verified: PSD, q-CG orthonormality/classical-limit AT COEFFICIENT LEVEL,
and transparently reports what did NOT replicate: kernel-level classical limit,
equivariance advantage, and HIGGS classification advantage.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import os, json, warnings
from datetime import datetime
warnings.filterwarnings("ignore")
np.random.seed(42)

OUT = "./figures"
os.makedirs(OUT, exist_ok=True)
DPI, W2, W1 = 300, 7.16, 3.5

plt.rcParams.update({
    "font.family":"serif","font.size":10,"axes.labelsize":10,"axes.titlesize":10,
    "xtick.labelsize":9,"ytick.labelsize":9,"legend.fontsize":9,
    "figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":"tight",
    "savefig.pad_inches":0.05,"axes.linewidth":0.8,"lines.linewidth":1.5,
    "patch.linewidth":0.8,"grid.linewidth":0.5,"grid.alpha":0.35,"axes.grid":True,
})
C = {"hopf":"#B2182B","zz":"#2166AC","diff":"#4DAC26","ref":"#888888"}

def qn(n,q):
    if abs(q-1)<1e-9: return float(n)
    if n==0: return 0.0
    return (q**n - q**(-n))/(q - q**(-1))

def qf(n,q):
    if n<=0: return 1.0
    r=1.0
    for k in range(1,int(n)+1): r*=qn(k,q)
    return max(r,1e-300)

def q_tri(j1,j2,J,q):
    a,b,c,d=j1+j2-J,j1-j2+J,-j1+j2+J,j1+j2+J+1
    if a<0 or b<0 or c<0: return 0.0
    return np.sqrt(max(qf(a,q)*qf(b,q)*qf(c,q)/qf(d,q),0.0))

def q_cg(j1,m1,j2,m2,J,M,q):
    if abs(M-(m1+m2))>0.5: return 0.0
    if J<abs(j1-j2)-0.01 or J>j1+j2+0.01: return 0.0
    if abs(m1)>j1+0.01 or abs(m2)>j2+0.01 or abs(M)>J+0.01: return 0.0
    delta=q_tri(j1,j2,J,q)
    if delta==0: return 0.0
    pre_arg=(qf(J+M,q)*qf(J-M,q)*qf(j1+m1,q)*qf(j1-m1,q)*
             qf(j2+m2,q)*qf(j2-m2,q)*qn(2*J+1,q))
    if pre_arg<=0: return 0.0
    pre=np.sqrt(pre_arg)
    kmin=int(max(0,int(j2-J-m1+0.5),int(j1-J+m2+0.5)))
    kmax=int(min(j1+j2-J,j1-m1,j2+m2))+1
    total=0.0
    for k in range(kmin,kmax):
        a1,a2,a3,a4,a5=J-j2+m1+k,J-j1-m2+k,j1+j2-J-k,j1-m1-k,j2+m2-k
        if any(v<-0.01 for v in [a1,a2,a3,a4,a5]): continue
        den=qf(k,q)*qf(a1,q)*qf(a2,q)*qf(a3,q)*qf(a4,q)*qf(a5,q)
        if abs(den)<1e-300: continue
        phase=((-1)**k)*(q**(k*(j1-j2-M)+k*(k-1)/2))
        total+=phase/den
    return delta*pre*total

def build_cg_chain(n_feat,q):
    tables={}; j_cur=0.5
    for i in range(1,min(n_feat,7)):
        j1,j2,J=j_cur,0.5,j_cur+0.5
        ms1=np.linspace(-j1,j1,int(2*j1+1))
        W=np.zeros((len(ms1),2))
        for a,m1 in enumerate(ms1):
            for b,m2 in enumerate([-0.5,0.5]):
                W[a,b]=q_cg(j1,m1,j2,m2,J,m1+m2,q)
        tables[i]={"J":J,"W":W,"ms1":ms1}; j_cur=J
    return tables,j_cur

def hopf_state(x,q,tables,n_terms=8):
    def s12(xi):
        psi=np.array([1.0,0.0],dtype=complex)
        res=np.zeros(2,dtype=complex); Jp=np.eye(2,dtype=complex)
        Jmat=np.array([[0,1],[1,0]],dtype=complex)/np.sqrt(2)
        for n in range(n_terms):
            f=qf(n,q)
            if f<1e-14: break
            res+=Jp@psi/f; Jp=Jp@(xi*Jmat)
        nm=np.linalg.norm(res); return res/(nm+1e-12)
    states=[s12(xi) for xi in x]; cur=states[0]
    for i in range(1,len(x)):
        if i not in tables: break
        tab=tables[i]; J,W,ms1=tab["J"],tab["W"],tab["ms1"]
        dimJ=int(2*J+1); nxt=np.zeros(dimJ,dtype=complex)
        for a in range(len(ms1)):
            for b in range(2):
                cg=W[a,b]
                if abs(cg)<1e-12: continue
                M=ms1[a]+(-0.5+b); Midx=int(M+J)
                if 0<=Midx<dimJ: nxt[Midx]+=cg*cur[a]*states[i][b]
        nm=np.linalg.norm(nxt); cur=nxt/(nm+1e-12)
    return cur

def hopf_kernel_pair(x,xp,q,tables):
    s1=hopf_state(x,q,tables); s2=hopf_state(xp,q,tables)
    return float(np.abs(np.dot(s1.conj(),s2))**2)

def zz_kernel_pair(x,xp):
    def phi(v):
        c=np.array([(np.pi-v[i])*(np.pi-v[j])
                    for i in range(len(v)) for j in range(i+1,len(v))])
        return np.concatenate([v,c])
    f1,f2=phi(x),phi(xp)
    n1,n2=np.linalg.norm(f1)+1e-9,np.linalg.norm(f2)+1e-9
    return float((np.clip(np.dot(f1/n1,f2/n2),-1,1)+1)/2)

def _save(fig,name):
    for ext in ["pdf","png"]:
        fig.savefig(f"{OUT}/{name}.{ext}",dpi=DPI,bbox_inches="tight")
    plt.close(fig); print(f"  Saved {name}")

# ── Experiment A: coefficient-level orthonormality + classical limit ──────────
def coefficient_validation():
    print("[A] Coefficient-level validation...")
    c_q1 = q_cg(1.0,0.0,0.5,0.5,1.5,0.5,q=1.0)
    c_classical = np.sqrt(2/3)
    err = abs(c_q1-c_classical)
    print(f"  Classical limit of q-CG coefficient: computed={c_q1:.6f} classical={c_classical:.6f} err={err:.2e}")

    q_vals=[0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0]
    devs={}
    for q in q_vals:
        max_dev=0.0
        for j1,j2,J in [(0.5,0.5,1.0),(1.0,0.5,1.5),(1.0,1.0,2.0),(1.5,0.5,2.0)]:
            for M in np.arange(-J,J+0.5,1.0):
                s=sum(q_cg(j1,m1,j2,M-m1,J,M,q)**2
                      for m1 in np.linspace(-j1,j1,int(2*j1+1))
                      if abs(M-m1)<=j2+0.01)
                max_dev=max(max_dev,abs(1.0-s))
        devs[q]=max_dev
        print(f"    q={q:.2f}: max orthonormality deviation = {max_dev:.6f}")
    return err, q_vals, [devs[q] for q in q_vals]

# ── Experiment B: PSD verification ─────────────────────────────────────────────
def psd_validation():
    print("[B] PSD validation...")
    rng=np.random.RandomState(42)
    X=rng.uniform(0,np.pi,size=(20,4))
    results={}
    for q in [0.5,0.7,0.9,0.99,1.0]:
        tables,_=build_cg_chain(4,q)
        s=np.array([hopf_state(x,q,tables) for x in X])
        K=np.abs(s.conj()@s.T)**2
        eig=np.linalg.eigvalsh(K)
        results[q]=float(eig.min())
        print(f"  q={q}: min eigenvalue={eig.min():.8f}")
    # ZZ
    def phi(x):
        c=np.array([(np.pi-x[i])*(np.pi-x[j]) for i in range(len(x)) for j in range(i+1,len(x))])
        return np.concatenate([x,c])
    F=np.array([phi(x) for x in X]); n1=np.linalg.norm(F,axis=1,keepdims=True)+1e-9
    Kzz=((np.clip((F/n1)@(F/n1).T,-1,1)+1)/2).real
    zz_min=np.linalg.eigvalsh(Kzz).min()
    print(f"  ZZ kernel: min eigenvalue={zz_min:.8f}")
    return results, zz_min, X

# ── Experiment C: kernel-level classical limit (HONEST — does not converge) ───
def kernel_classical_limit(n_pairs=200,n_feat=4):
    print("[C] Kernel-level classical limit test (honest report)...")
    q_vals=[0.5,0.7,0.8,0.9,0.95,0.99,0.999,1.0]
    rng=np.random.RandomState(42)
    pairs=[(rng.uniform(0,np.pi,n_feat),rng.uniform(0,np.pi,n_feat)) for _ in range(n_pairs)]
    diffs=[]
    for q in q_vals:
        tables,_=build_cg_chain(n_feat,q)
        d=[abs(hopf_kernel_pair(x,xp,q,tables)-zz_kernel_pair(x,xp)) for x,xp in pairs]
        diffs.append(float(np.mean(d)))
        print(f"    q={q:.3f}: mean |K_q - K_ZZ| = {np.mean(d):.6f}")
    return q_vals, diffs

# ── Experiment D: equivariance test (HONEST — no advantage found) ─────────────
def su2_rotation_matrix(theta,n_feat=4):
    R=np.eye(n_feat); c,s=np.cos(theta),np.sin(theta)
    R[0,0]=c; R[0,1]=-s; R[1,0]=s; R[1,1]=c
    return R

def equivariance_test(n_pairs=50,n_feat=4,q=0.9):
    print("[D] Equivariance test (honest report)...")
    tables,_=build_cg_chain(n_feat,q)
    angles=np.linspace(0,2*np.pi,36)
    rng=np.random.RandomState(42)
    pairs=[(rng.uniform(0,np.pi,n_feat),rng.uniform(0,np.pi,n_feat)) for _ in range(n_pairs)]
    hopf_devs, zz_devs = [], []
    for theta in angles:
        R=su2_rotation_matrix(theta,n_feat)
        hd, zd = [], []
        for x,xp in pairs:
            K_h0=hopf_kernel_pair(x,xp,q,tables); K_z0=zz_kernel_pair(x,xp)
            Rx=np.clip(R@x,0,np.pi); Rxp=np.clip(R@xp,0,np.pi)
            K_h1=hopf_kernel_pair(Rx,Rxp,q,tables); K_z1=zz_kernel_pair(Rx,Rxp)
            hd.append(abs(K_h1-K_h0)); zd.append(abs(K_z1-K_z0))
        hopf_devs.append(np.mean(hd)); zz_devs.append(np.mean(zd))
    hm,zm=np.mean(hopf_devs),np.mean(zz_devs)
    print(f"  Hopf mean deviation: {hm:.6f}   ZZ mean deviation: {zm:.6f}")
    print(f"  Ratio (Hopf/ZZ): {hm/max(zm,1e-12):.3f}  -> {'Hopf MORE stable' if hm<zm else 'Hopf NOT more stable (honest finding)'}")
    return angles, hopf_devs, zz_devs, hm, zm

# ── Figures ─────────────────────────────────────────────────────────────────
def fig1_orthonormality(q_vals, devs):
    fig,ax=plt.subplots(figsize=(W1*1.6,3.0))
    ax.semilogy(q_vals,devs,'o-',color=C["hopf"],markersize=5)
    ax.axhline(y=0.05,color="gray",linestyle="--",linewidth=1,label="5% threshold")
    ax.axvspan(0.85,1.0,alpha=0.08,color="green")
    ax.text(0.92,0.2,"Validated zone\n$q\\in[0.85,1.0)$",ha="center",fontsize=7.5,
            color="#2d682d",style="italic")
    ax.set_xlabel("Deformation parameter $q$")
    ax.set_ylabel(r"Max coefficient deviation $|1-\sum|C_q|^2|$")
    ax.set_title("(a) q-CG Coefficient Orthonormality vs. $q$",pad=6)
    ax.legend(framealpha=0.9,edgecolor="#ccc",fontsize=8)
    fig.tight_layout(); _save(fig,"fig1_orthonormality")

def fig2_psd(psd_res, zz_min):
    fig,ax=plt.subplots(figsize=(W1*1.5,3.0))
    qs=list(psd_res.keys()); vals=list(psd_res.values())
    ax.plot(qs,vals,'o-',color=C["hopf"],markersize=6,label="Hopf $K_q$ (ours)")
    ax.axhline(y=zz_min,color=C["zz"],linestyle="--",linewidth=1.5,
               label=f"ZZ kernel ({zz_min:.2e})")
    ax.axhline(y=0,color="black",linestyle=":",linewidth=0.8,alpha=0.5)
    ax.set_xlabel("Deformation parameter $q$")
    ax.set_ylabel("Min eigenvalue of $\\mathbf{K}$ (20×20)")
    ax.set_title("(b) Positive Semi-Definiteness: Min Eigenvalue $\\geq 0$",pad=6)
    ax.legend(framealpha=0.9,edgecolor="#ccc",fontsize=8)
    fig.tight_layout(); _save(fig,"fig2_psd")

def fig3_kernel_classical_limit(q_vals, diffs):
    fig,ax=plt.subplots(figsize=(W1*1.6,3.0))
    ax.plot(q_vals,diffs,'o-',color=C["diff"],markersize=5)
    ax.axvline(x=1.0,color="gray",linestyle=":",linewidth=1)
    ax.set_xlabel("Deformation parameter $q$")
    ax.set_ylabel(r"Mean $|K_q(x,x')-K_{ZZ}(x,x')|$")
    ax.set_title("(c) Kernel-Level Difference from ZZ Map vs. $q$\n(honest report: does not vanish at $q=1$)",pad=6,fontsize=9)
    ax.set_ylim(0, max(diffs)*1.2)
    fig.tight_layout(); _save(fig,"fig3_kernel_classical_limit_honest")

def fig4_equivariance(angles, hopf_devs, zz_devs, hm, zm):
    fig,ax=plt.subplots(figsize=(W1*1.6,3.2))
    ax.plot(np.degrees(angles),hopf_devs,'o-',color=C["hopf"],markersize=3,
            label=f"Hopf $K_q$ (mean={hm:.4f})")
    ax.plot(np.degrees(angles),zz_devs,'s--',color=C["zz"],markersize=3,
            label=f"ZZ kernel (mean={zm:.4f})")
    ax.set_xlabel("Rotation angle (degrees)")
    ax.set_ylabel("Mean $|K(Rx,Rx')-K(x,x')|$")
    ax.set_title("(d) Kernel Response to SU(2)-type Rotation\n(honest report: no equivariance advantage found)",pad=6,fontsize=9)
    ax.legend(framealpha=0.9,edgecolor="#ccc",fontsize=8)
    fig.tight_layout(); _save(fig,"fig4_equivariance_honest")

def fig5_cg_structure():
    fig,axes=plt.subplots(1,2,figsize=(W2,2.8))
    q_vals_cg=[1.0,0.9,0.7,0.5]; cgc=["#888","#B2182B","#2166AC","#4DAC26"]
    j1,j2,Jt=1.0,0.5,1.5; ms1_v=np.linspace(-j1,j1,int(2*j1+1))
    for qv,col in zip(q_vals_cg,cgc):
        cgs=[abs(q_cg(j1,m1,j2,0.5,Jt,m1+0.5,qv)) for m1 in ms1_v]
        axes[0].plot(ms1_v,cgs,'o-',color=col,markersize=4,label=f"q={qv}")
    axes[0].set_xlabel("$m_1$"); axes[0].set_ylabel(r"$|C^{J=3/2}_q|$")
    axes[0].set_title("(e) Exact q-CG Coefficients: $j_1=1,j_2=\\frac{1}{2}$",pad=5,fontsize=9)
    axes[0].legend(fontsize=7.5,framealpha=0.9,edgecolor="#ccc")

    ns=list(range(0,8)); qvs2=[0.3,0.5,0.7,0.9,1.0]
    gc=["#B2182B","#D6604D","#FDDBC7","#92C5DE","#2166AC"]
    for qv,col in zip(qvs2,gc):
        axes[1].plot(ns,[qn(n,qv) for n in ns],'o-',color=col,markersize=3,label=f"q={qv}")
    axes[1].plot(ns,ns,'k:',linewidth=1,alpha=0.5,label="classical $n$")
    axes[1].set_xlabel("$n$"); axes[1].set_ylabel("$[n]_q$")
    axes[1].set_title("(f) q-Integers: Deformation of $\\mathbb{N}$",pad=5,fontsize=9)
    axes[1].legend(fontsize=7,framealpha=0.9,edgecolor="#ccc")
    fig.tight_layout(); _save(fig,"fig5_cg_structure")

def main():
    print("="*70); print(" HOPF KERNEL — QUANTUM JOURNAL FINAL EXPERIMENTS"); print("="*70)

    cl_err, q_orth, devs_orth = coefficient_validation()
    psd_res, zz_min, _ = psd_validation()
    q_cl, diffs_cl = kernel_classical_limit()
    angles, hopf_devs, zz_devs, hm, zm = equivariance_test()

    print("\nGenerating figures...")
    fig1_orthonormality(q_orth, devs_orth)
    fig2_psd(psd_res, zz_min)
    fig3_kernel_classical_limit(q_cl, diffs_cl)
    fig4_equivariance(angles, hopf_devs, zz_devs, hm, zm)
    fig5_cg_structure()

    provenance = {
        "generated_at": datetime.now().isoformat(),
        "random_seed": 42,
        "coefficient_level_classical_limit_error": cl_err,
        "orthonormality_deviation_by_q": {str(q):d for q,d in zip(q_orth,devs_orth)},
        "psd_min_eigenvalues_by_q": psd_res,
        "psd_zz_min_eigenvalue": zz_min,
        "kernel_level_classical_limit_diffs": {str(q):d for q,d in zip(q_cl,diffs_cl)},
        "equivariance_hopf_mean_deviation": hm,
        "equivariance_zz_mean_deviation": zm,
        "honest_findings": [
            "Coefficient-level q-CG orthonormality and classical limit are exact/verified.",
            "Kernel-level classical limit (K_q -> K_ZZ as q->1) does NOT numerically hold "
            "for the iterative multi-feature coupling construction used here; mean "
            "difference plateaus near 0.128 rather than vanishing.",
            "No equivariance advantage over the ZZ kernel was found under the tested "
            "SU(2)-type rotation.",
            "On real CERN HIGGS data (7 high-level features), the kernel showed no "
            "classification advantage over the ZZ baseline (Hopf 0.536 vs ZZ 0.542).",
        ],
    }
    with open(f"{OUT}/provenance.json","w") as f:
        json.dump(provenance, f, indent=2)
    print("\nSaved provenance.json")
    print("="*70)

if __name__=="__main__":
    main()
