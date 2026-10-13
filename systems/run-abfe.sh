
window='coulomb'

# Set some environment variables 
FREE_ENERGY=`pwd`
echo "Free energy home directory set to $FREE_ENERGY"
mdps="../mdp_templates/${window}"

get_mdp() {
    if [[ $# -ne 2 ]]; then
        echo "get_mdp: Usage: get_mdp input.mdp index" >&2
        return 1
    fi

    local mdp="$1"
    local idx="$2"

    [[ -f $mdp ]] || { echo "get_mdp: Error: '$mdp' not found." >&2; return 1; }

    local filename base out
    filename=$(basename -- "$mdp")      
    base="${filename%%.*}"              
    out="${base}_${idx}.mdp"            

    # Rewrite only the init-lambda-state line; keep everything else unchanged.
    awk -v idx="$idx" '
        BEGIN { found = 0 }
        /^[[:space:]]*init[-_]lambda[-_]state[[:space:]]*=/ {
            sub(/=.*/, "= " idx)
            found = 1
        }
        { print }
        END {
            if (!found)
                print "Warning: init-lambda-state not found in the template." \
                      > "/dev/stderr"
        }
    ' "$mdp" > "$out"

    echo "Wrote: $out"
}


for LAMBDA in $(seq 0 10); do
    mkdir ${window}_$LAMBDA
    cd ${window}_$LAMBDA

    #######################
    # ENERGY MINIMIZATION #
    #######################
    echo "Starting sd-minimization for lambda = $LAMBDA..." 

    # Iterative calls to grompp and mdrun to run the simulations
    get_mdp ${mdps}/em.mdp $LAMBDA
    gmx grompp -f em_$LAMBDA.mdp -c $FREE_ENERGY/solv.gro -p $FREE_ENERGY/topol.top -o em_$LAMBDA.tpr -maxwarn 3 

    gmx mdrun -deffnm em_$LAMBDA 

    sleep 3
    
    
    #######################
    # NVT EQUILIBRATION #
    #######################
    echo "Starting constant volume equilibration ..."
    
    get_mdp ${mdps}/nvt.mdp $LAMBDA
    gmx grompp -f nvt_$LAMBDA.mdp -c em_$LAMBDA.gro -r em_$LAMBDA.gro -p $FREE_ENERGY/topol.top -o nvt_$LAMBDA.tpr -maxwarn 3

    gmx mdrun -deffnm nvt_$LAMBDA

    echo "Constant volume equilibration complete."

    sleep 3

    #############################
    # NPT EQUILIBRATION-RESTR #
    #############################
    echo "Starting constant pressure equilibration ..."
    get_mdp ${mdps}/npt-restr.mdp $LAMBDA
    gmx grompp -f npt-restr_$LAMBDA.mdp -c nvt_$LAMBDA.gro -p $FREE_ENERGY/topol.top -t nvt_$LAMBDA.cpt -r nvt_$LAMBDA.gro -o npt-restr_$LAMBDA.tpr -maxwarn 3

    gmx mdrun -deffnm npt-restr_$LAMBDA

    echo "Constant pressure equilibration complete."

    sleep 3

    
    #####################
    # NPT EQUILIBRATION #
    #####################
    echo "Starting constant pressure equilibration..."

    get_mdp ${mdps}/npt.mdp $LAMBDA
    gmx grompp -f npt_$LAMBDA.mdp -c npt-restr_$LAMBDA.gro -p $FREE_ENERGY/topol.top -t npt-restr_$LAMBDA.cpt -r npt-restr_$LAMBDA.gro -o npt_$LAMBDA.tpr -maxwarn 3

    gmx mdrun -deffnm npt_$LAMBDA

    echo "Constant pressure equilibration complete."

    sleep 3


    #################
    # PRODUCTION MD #
    #################
    echo "Starting production MD simulation..."

    get_mdp ${mdps}/md.mdp $LAMBDA
    gmx grompp -f md_$LAMBDA.mdp -c npt_$LAMBDA.gro -p $FREE_ENERGY/topol.top -t npt_$LAMBDA.cpt -o md_$LAMBDA.tpr -maxwarn 3

    gmx mdrun -deffnm md_$LAMBDA -dhdl dhdl_$LAMBDA.xvg

    echo "Production MD complete."

    # End
    echo "Ending. Job completed for lambda = $LAMBDA"

    cd $FREE_ENERGY
done
exit;
