"""
    Struct: ForkTensorNetworkOperator
    Representing the Fork Tensor Network Operator
    Copyright (C) 2024 Hyun-Yong Lee <hyunyong@korea.ac.kr>
"""
struct ForkTensorNetworkOperator

    Lx::Int                               # Number of Sites in the x-direction
    Ly::Int                               # Number of Sites in the y-direction
    phys_idx::Matrix{Index}               # Physical Index
    aux_x_idx::Vector{Index}              # Auxiliary Index in the x-direction
    aux_y_idx::Matrix{Index}              # Auxiliary Index in the y-direction
    Ws::Matrix{ITensor}                   # List of Tensors



    """
        ForkTensorNetworkOperator(Ws::Matrix{ITensor}, phys_idx::Matrix{Index}, aux_x_idx::Vector{Index}, aux_y_idx::Matrix{Index})

        The constructor of the ForkTensorNetworkOperator struct.
    """
    function ForkTensorNetworkOperator(Ws::Matrix{ITensor}, phys_idx::Matrix{Index}, aux_x_idx::Vector{Index}, aux_y_idx::Matrix{Index})
        Lx = size(Ws, 1)
        Ly = size(Ws, 2)
        Ĥ = new(
            Lx,                            # Number of Sites in the x-direction
            Ly,                            # Number of Sites in the y-direction
            phys_idx,                      # Physical Index
            aux_x_idx,                     # Auxiliary Index in the x-direction
            aux_y_idx,                     # Auxiliary Index in the y-direction
            Ws                             # List of Tensors
        )
        return Ĥ
    end

end # struct ForkTensorNetworkOperator


"""
    flux_check(Ĥ::ForkTensorNetworkOperator)

    Check the flux of the ForkTensorNetworkOperator.
"""
function flux_check(Ĥ::ForkTensorNetworkOperator)

    # Show error if the tensor is not conserving qn
    hasqns(Ĥ.phys_idx[1, 1]) != true && throw(ArgumentError("The QN is not conserved"))

    println("Checking Flux...")
    println("Lx = $(Ĥ.Lx), Ly = $(Ĥ.Ly)")

    for x = 1:Ĥ.Lx
        for y = 1:Ĥ.Ly

            if checkflux(Ĥ.Ws[x, y]) === nothing
                println("x = $(x), y = $(y): ", flux(Ĥ.Ws[x, y]))
            end
        end
    end

end
