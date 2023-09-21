build_base() {
    echo "Deleting base container"
    rm containers/base.sif 
    echo "building base"
    apptainer build --force containers/base.sif containers/base.def
}

remove_lab(){
    echo "Deleting lab container"
    rm containers/lab.sif 
}

build_sandbox(){
    remove_lab
    echo "building sandbox lab"
    sudo apptainer build --sandbox --fix-perms containers/lab.sif containers/lab.def
}

build_lab(){
    remove_lab
    echo "building lab"
    apptainer build --force containers/lab.sif containers/lab.def
}

read -p "Are you sure you want to rebuild $1? (Y/n): " answer


# Check the user's response
if [[ $answer == "Y" ]]; then
    case "$1" in
    "base")
        echo "Building base"
        build_base
        ;;
    "sandbox")
        echo "Building sandbox"
        build_sandbox
        ;;
    "lab")
        echo "Building lab"
        build_lab
        ;;
    *)
        echo "Building lab image from scratch"
        build_base
        build_lab
        ;;
    esac
else
    echo "Build not executed."
fi

