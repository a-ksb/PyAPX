using Combinatorics
using ProgressMeter

# Generate the next permutation
function next_permutation!(a)
    n = length(a)
    i = findlast(i -> a[i] < a[i+1], 1:n-1)
    i === nothing && return false

    j = findlast(j -> a[j] > a[i], i+1:n)
    j === nothing && return false
    j += i

    a[i], a[j] = a[j], a[i]
    reverse!(@view a[i+1:end])
    return true
end

# Generate permutations and write to candidates.csv
function gen_permutations_to_csv(arr, filename)
    sorted_arr = sort(arr)
    structure_id = 0

    open(filename, "w") do file
        # Write CSV header
        header = ["structure_id"]
        for i in 1:18
            push!(header, "site_$i")
        end
        println(file, join(header, ","))

        # Write the first permutation (structure_id = 0)
        println(file, "$structure_id," * "C," * join(sorted_arr, ","))

        # Generate and write the next permutation
        while next_permutation!(sorted_arr)
            structure_id += 1
            println(file, "$structure_id," * "C," * join(sorted_arr, ","))
        end
    end
    
    return structure_id + 1
end

# Create input array (17 atoms, C is fixed at the first position)
input_array = ["B", "B", "B", "B", "B", "B", "C", "C", "C", "C", "C", "N", "N", "N", "N", "N", "N"]

total_structures = gen_permutations_to_csv(input_array, "candidates.csv")

println("finish generate candidates.csv")
println("Total number of structures: $total_structures")
