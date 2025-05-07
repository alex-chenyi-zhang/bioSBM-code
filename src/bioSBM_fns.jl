using Random, Distributions, StatsBase, LinearAlgebra, DelimitedFiles, Optim, LineSearches, Flux
using JLD2  ## this is to store trained flux model to be loaded for future usage
Random.seed!(1)

################################################################################

# This function computes the approximate ELBO of the model with the gaussian likelihood
function ELBO_gauss(ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    Σ::Array{Float64, 2}, B::Array{Float64, 2}, μ, Y::Array{Float64, 2}, K::Int, N::Int, like_var::Array{Float64, 2})
    apprELBO = 0.0
    inv_Σ = inv(Σ)
    #inv_Σ = Matrix(1.0I, length(Σ[1,:]), length(Σ[1,:]))
    apprELBO = 0.5 * N * log(det(inv_Σ))
    for i in 1:N
        apprELBO -= 0.5 * (dot(λ[:,i] .- μ[:,i], inv_Σ, λ[:,i] .- μ[:,i]) + tr(inv_Σ*ν[:,:,i])) #first 2 terms
    end

    if isnan(apprELBO)
        println("ERROR 1")
        #break
    end

    for k in 1:K
        for j in 1:N
            for i in 1:N
                if i != j && ϕ[i,j,k] > eps()
                    apprELBO -= ϕ[i,j,k]*log(ϕ[i,j,k]) #last entropic term
                end
            end
        end
    end
    if isnan(apprELBO)
        println("ERROR 2")
        #break
    end

    @views for j in 1:N
        for i in 1:N
            if i != j
                apprELBO += dot(ϕ[i,j,:],λ[:,i])
            end
        end
    end
    if isnan(apprELBO)
        println("ERROR 3")
        #break
    end

    theta = zeros(K)
    for i in 1:N
        theta .= softmax(λ[:,i])
        # second line of the expression above
        apprELBO -= (N-1) * ( (log(sum(exp.(λ[:,i])))) + 0.5*tr((diagm(theta) .- theta * theta')*ν[:,:,i]) )
        #gaussian entropic term
        apprELBO += 0.5*log(det(ν[:,:,i]))
    end
    if isnan(apprELBO)
        println("ERROR 4")
        #break
    end
    #println("Partial ELBO: ", apprELBO)
    #likelihood term
    inv_like_var = 0.5 ./ like_var;
    log_like_var = 0.5 .* log.(like_var);

    for i in 1:N
        for j in 1:i-1
            for k in 1:K
                for g in 1:K
                    logP = (-(Y[i,j] - B[k,g])^2 * inv_like_var[k,g] - log_like_var[k,g]) 
                    apprELBO += ϕ[i,j,k]*ϕ[j,i,g]*logP
                end
            end
        end
    end
    #apprELBO += -0.25*N*(N-1)*log(like_var[1])
    if isnan(apprELBO)
        println("ERROR 5")
        #break
    end
    return apprELBO
end


#########################################################################################
#########################################################################################
#########################################################################################

#=
In the E-step due to the non-conjugancy of the logistic normal with the multinomial
we resort to a gaussian approximation of the variational porsterior (Wang, Blei, 2013).
The approximations equals an optimizetion process that we characterize with the followfin functions
=#

function f(η_i::Array{Float64, 1}, ϕ_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i, N::Int)
    #f = 0.5 * dot(η_i .- μ_i, inv_Σ, η_i .- μ_i) - dot(η_i, ϕ_i) +(N-1)*log(sum(exp.(η_i)))
    f = 0.5 * dot(η_i .- μ_i, inv_Σ, η_i .- μ_i) - dot(η_i, ϕ_i) +(N-1)*log(sum(exp.(η_i)))

    return f
end

# gradient of f
function gradf!(G, η_i::Array{Float64, 1}, ϕ_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i, N::Int)
    G .= softmax(η_i)*(N-1) .- ϕ_i .+ inv_Σ*(η_i .- μ_i)

end

# Hessian of f
function hessf!(H, η_i::Array{Float64, 1}, inv_Σ::Array{Float64, 2}, μ_i, N::Int)
    #theta = exp.(η_i)/sum(exp.(η_i))
    theta = softmax(η_i)
    H .=  (N-1)*(diagm(theta) .- theta*theta') .+ inv_Σ
end


################################################################################
# Function that perform the variational optimization
function Estep_logitNorm!(ϕ::Array{Float64, 3}, λ, ν::Array{Float64, 3},
    inv_Σ::Array{Float64, 2}, μ, N::Int, K::Int)
    G = zeros(K)
    H = zeros(K,K)
    for i in 1:N
        ϕ_i = sum(@view(ϕ[i,:,:]),dims=1)[1,:]
        μ_i = @view μ[:,i]
        #res = optimize(η_i -> f(η_i, ϕ_i, inv_Σ, μ_i, N), (G, η_i) -> gradf!(G,η_i, ϕ_i, inv_Σ, μ_i, N), randn(K), BFGS(linesearch = LineSearches.BackTracking(order=2)))#BFGS())
        res = optimize(η_i -> f(η_i, ϕ_i, inv_Σ, μ_i, N), (G, η_i) -> gradf!(G,η_i, ϕ_i, inv_Σ, μ_i, N), λ[:, i], BFGS(linesearch = LineSearches.BackTracking(order=2)))#BFGS())
        η_i = Optim.minimizer(res)
        hessf!(H, η_i, inv_Σ, μ_i, N)
        λ[:,i] .= η_i
        ν[:,:,i] .= Hermitian(inv(H))
    end
end


# In-place softmax function
function softmax!(x)
    max_x = maximum(x)
    @inbounds for i in eachindex(x)
        x[i] = exp(x[i] - max_x)
    end
    sum_x = sum(x)
    @inbounds for i in eachindex(x)
        x[i] /= sum_x
    end
end

function Estep_multinomial_gauss!(ϕ::Array{Float64, 3}, λ, B::Array{Float64, 2},
    Y::Array{Float64, 2}, N::Int, K::Int, like_var::Array{Float64, 2})
    
    # Precompute reusable quantities
    inv_like_var = 0.5 ./ like_var
    log_like_var = 0.5 .* log.(like_var)
    
    # for i in 1:N
   for i in sample(1:N, div(N,4), replace=false)
        # for j in 1:N
       for j in sample(1:N, div(N,4), replace=false)
            if i != j
                for k in 1:K
                    logPi = λ[k,i]
                    for g in 1:K
                        logPi += -ϕ[j,i,g] *( ((Y[i,j] - B[k,g])^2) * inv_like_var[k,g]  + log_like_var[k,g])
                    end
                    ϕ[i,j,k] = logPi
                end
                # ϕ[i,j,:] = softmax(ϕ[i,j,:])
                softmax!(view(ϕ, i, j, :))
            end
        end
    end
end




#########################################################################################
#########################################################################################

function Mstep_blockmodel_gauss_multi!(ϕ::Vector{Array{Float64, 3}}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    Y::Vector{Array{Float64, 2}}, Ns::Array{Int,1}, K::Int, n_regions::Int)
    lv = 0.
    learn_r = 0.1
    cum_den = 0.
    for k in 1:K
        for g in 1:k  ###  !!! small k
            num_gauss = 0.
            num = 0.
            den = 0.
            for i_region in 1:n_regions
                @inbounds for j in 1:Ns[i_region]
                    @inbounds for i in 1:Ns[i_region]
                # for j in sample(1:Ns[i_region], div(Ns[i_region], 4))
                #     for i in sample(1:Ns[i_region], div(Ns[i_region], 4))
                        phi_prod = ϕ[i_region][i,j,k]*ϕ[i_region][j,i,g]
                        num += phi_prod*Y[i_region][i,j]
                        den += phi_prod
                        num_gauss += phi_prod * (Y[i_region][i,j] - B[k,g])^2
                        #lv  += phi_prod * (Y[i,j] - B[k,g])^2
                    end
                end
            end
            B[k,g] =  (1-learn_r)*B[k,g] + learn_r*num/(den)
            #cum_den += den
            like_var[k,g] =  (1-learn_r)*like_var[k,g] + learn_r*num_gauss/(den)
            B[g,k] = B[k,g]
            like_var[g,k] = like_var[k,g]
        end
    end

end





#########################################################################################
#########################################################################################
#########################################################################################


function run_VEM_gauss_NN!(max_n_iterations::Int, ϕ::Vector{Array{Float64, 3}}, λ, ν::Vector{Array{Float64, 3}},
    Σ::Array{Float64, 2}, B::Array{Float64, 2}, like_var::Array{Float64, 2},
    μ::Array{Float64, 2}, Y::Vector{Array{Float64, 2}}, X::Array{Float64, 2}, Γ, ps, K::Int, Ns::Array{Int,1}, P::Int, n_regions::Int, R::Float64)

    Ncum = 0
    N_s = ones(Int, n_regions)
    N_e = ones(Int, n_regions)
    for i_region in 1:n_regions
        N_s[i_region] = Ncum + 1
        N_e[i_region] = Ncum + Ns[i_region]
        Ncum += Ns[i_region]
    end

    ### elbows = zeros(n_regions, n_iterations)
    n_skip = 20
    elbows = zeros(n_regions, div(max_n_iterations,n_skip)+1)
    det_Sigma = zeros(div(max_n_iterations,n_skip)+1)
    #det_nu = [zeros(N, n_iterations) for i_reg in 1:n_regions]
    # opt = ADAM(0.01) #the value in the brackets is
    opt = Descent(0.01)
    #################################
    # definition of the loss functional to be used to optimize the flux model
    L(a,b) = (Flux.Losses.kldivergence(softmax(Γ(a)), b))

    #################################
    prev_loss = Inf
    tolerance = 1e-4
    max_flux_iters = 50

    elbo_eps = 5e-4
    elbo_0 = -Inf
    finish_iter = 1

    for i_iter in 1:max_n_iterations
        inv_Σ = inv(Σ)
        for i_region in 1:n_regions
            Estep_logitNorm!(ϕ[i_region], @view(λ[:,N_s[i_region]:N_e[i_region]]), ν[i_region], inv_Σ, Float64.(μ[:,N_s[i_region]:N_e[i_region]]), Ns[i_region], K)
            for m in 1:5
                Estep_multinomial_gauss!(ϕ[i_region], @view(λ[:,N_s[i_region]:N_e[i_region]]), B, Y[i_region], Ns[i_region], K, like_var)
            end
            Estep_logitNorm!(ϕ[i_region], @view(λ[:,N_s[i_region]:N_e[i_region]]), ν[i_region], inv_Σ, Float64.(μ[:,N_s[i_region]:N_e[i_region]]), Ns[i_region], K)

        end

        #n_flux = 40
        softmax_lamb = softmax(λ)
        for i_flux in 1:max_flux_iters  #n_flux
            gs = gradient(()-> L(X, softmax_lamb), ps)
            Flux.Optimise.update!(opt, ps, gs)
            current_loss = L(X, softmax_lamb)
            rel_change = abs(current_loss - prev_loss) / abs(prev_loss)

            if rel_change < tolerance
                # println("Early stopping at iteration $i_flux with relative change $rel_change")
                break
            end
            prev_loss = current_loss
        end

        μ = Γ(X);



        RΣ = zeros(K,K)
        for i_region in 1:n_regions
            for i in 1:Ns[i_region]
                RΣ .+= (ν[i_region][:,:,i] .+ (λ[:,i+N_s[i_region]-1] .- μ[:,i+N_s[i_region]-1])*(λ[:,i+N_s[i_region]-1] .- μ[:,i+N_s[i_region]-1])')
            end
            # println(RΣ)
        end
        RΣ .= sqrt(8*R*RΣ/(Ncum) + Matrix(1.0I, K, K)) .- Matrix(1.0I, K, K)
        Σ .= Hermitian(RΣ/(4*R))
        
        
        Mstep_blockmodel_gauss_multi!(ϕ, B, like_var, Y, Ns, K, n_regions)
        if i_iter == 1 || i_iter % n_skip == 0
            println("iter num: ", i_iter, "\n")
            for i_region in 1:n_regions
                elbows[i_region, div(i_iter,n_skip)+1] = -R*Ns[i_region]*tr(Σ) + ELBO_gauss(ϕ[i_region], λ[:,N_s[i_region]:N_e[i_region]], ν[i_region], Σ, B, μ[:,N_s[i_region]:N_e[i_region]],  Y[i_region], K, Ns[i_region], like_var)
                println("region: ", i_region," ELBO  \n", elbows[i_region,div(i_iter,n_skip)+1])
                if isnan(elbows[i_region, div(i_iter,n_skip)+1])
                    break
                end
            end

            det_Sigma[div(i_iter,n_skip)+1] = det(Σ)

            current_elbo = sum(elbows[:, div(i_iter,n_skip)+1])
            rel_change_elbo = abs(current_elbo - elbo_0) / abs(elbo_0)
            println("Rel.chang elbo: ", rel_change_elbo, "\n")
            println("Total elbo: ", current_elbo)


            if rel_change_elbo < elbo_eps
                println("VEM stoped at iteration $i_iter with relative change $rel_change_elbo")
                break
            end
            elbo_0 = current_elbo
            finish_iter = div(i_iter,n_skip)+1
        end
        
    end
    return elbows[:,1:finish_iter], det_Sigma[1:finish_iter] #, det_nu

end

##########################################################################################