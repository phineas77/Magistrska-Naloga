# Magistrska-Naloga
# Eksperimenti z hankelovo aproksimacijo majhnega ranga

To skladišče vsebuje kodo v jeziku **Julia** za eksperimente z **hankelovo aproksimacijo majhnega ranga** in sorodnimi metodami, zlasti:

- klasična MSSA / hankelova nizko-rangovna aproksimacija za **sintetične 1D signale** (`LR/`),
- **dušene** nizko-rangovne metode (`DLR/`),
- **adaptivna Hankel Low-Rank (HLR)** filtracija z izbiro ranga preko polinoma (tip Wang–Zhu) in rekonstrukcija s POCS (`HLR/`),
- `(Q,R)`-uteženi nizko-rangovni algoritem za časovno vrsto **Australian fortified wine** (`QR5.jl`),
- sintetični / seizmični eksperimenti v 5 dimenzijah.

Koda je nastala v okviru magistrskega dela o podatkih hankelovega tipa, odstranjevanju šuma in rekonstrukciji časovnih vrst.

---

## Odvisnosti

Večina skript uporablja podmnožico naslednjih paketov:

- `LinearAlgebra`
- `Statistics`
- `Random`
- `Plots`
- `Measures`
- `Printf`
- `LaTeXStrings`

Nekateri dodatno potrebujejo:

- `CSV`
- `DataFrames`
- `Dates`
- `JuMP`
- `OSQP`

Namestitev (v REPL):

```julia
using Pkg
Pkg.add([
    "LinearAlgebra",
    "Statistics",
    "Random",
    "Plots",
    "Measures",
    "Printf",
    "LaTeXStrings",
    "CSV",
    "DataFrames",
    "Dates",
    "JuMP",
    "OSQP",
])
