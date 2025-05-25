import numpy as np
import tenseal as ts

_context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60,40,40,60]
        )
_context.global_scale = 2 ** 40
_context.generate_galois_keys()

plaintext_vector = np.array([1.0, 1.0, 1.0, 1.0])
result = np.array([0, 0, 0, 0])
print("Original vector:", plaintext_vector)


enc_vector = ts.ckks_vector(_context, plaintext_vector)
enc_begin = enc_vector.decrypt()
max_err = np.max(np.abs(enc_begin - plaintext_vector))
enc_result = ts.ckks_vector(_context, result)


iter = 200000
for i in range(iter):
    if i % 2 == 0:
        enc_result += enc_vector
    else:
        enc_result -= enc_vector

# Decrypt the result
decrypted_result = enc_result.decrypt()
print("Decrypted result after addition:", decrypted_result)

# Calculate the expected result and error
expected_result = plaintext_vector * 0
max_error = max(np.abs(decrypted_result))

print(max_error)
print(max_err)