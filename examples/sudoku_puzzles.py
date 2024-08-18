# A dictionary with examples of sudoku puzzles

# The selected puzzles are a subset of the puzzles from the research paper by S.W. Wang: 
# “A Dataset of Sudoku Puzzles with Difficulty Metrics Experienced by Human Players,” in IEEE Access, 2024.
# This dataset is available at: https://github.com/synnwang/sudoku_dataset_difficulty

example_sudokus = {
#   Game No.    Sudoku Puzzle
    "124" :     "..7........5.4..7..695...31...4.58.2.5..2..4.6.23.1...29...358..3..1.2........3..",
    "388" :     "82..7.....95.86...6...4..3...6..8...13.....46...3..1...7..3...4...75.31.....9..85",
    "401" :     "..29...7..1...7....9.821...1.....5.7.47.6.21.2.9.....4...418.3....3...8..3...24..",
    "430" :     "2.163.48....9...3.3.....6......7..4651..8..7374..9......4.....7.3...5....79.628.4",
    "570" :     "8.....9...6.....84..47....69..46..7..389.754..7..38..97....28..24.....5...6.....2",
    "624" :     "...6.3..5.8.....3719..5....2.5....6..61...84..3....5.1....2..9897.....5.6..5.7...",
    "888" :     "...4.23.1........6..7.9.2.47..64.8....85.91....4.71..24.6.5.7..9........8.52.7...",
    "1135" :    "..35.7.64.9....7.......6.93....8..518.2...3.775..6....62.3.......5....3.34.2.51..",
    "1246" :    ".3.5.8.....9....3..58.762...9....4.86..9.4..38.4....2...572.16..2....3.....6.5.8.",
    "1342" :    "....1327......26..37.....8..13....6.9..3.8..2.8....73..3.....15..16......2719....",
    "1404" :    "93........654..9.84.7..9......87..5...8...6...7..14......7..4.67.3..681........27",
    "1411" :    ".56.9.17.1...7...6...1....2..9.128....8...2....168.5..3....1...2...4...9.65.2.43.",
    "1547" :    "36.25.84..5.9..6.........9..21.8....5.......8....9.42..3.........4..6.8..98.34.61",
    "1623" :    "89.7..1.5....6.....1.5..9.....3...6...9...8...2...5.....5..6.4.....2....6.4..3.21",
    "1705" :    "..9....5.6.573..9..1.6..2.4....9.6...8..4..2...3.8....2.6..3.4..7..623.9.3....8..",
    "1733" :    "5....1.73..86....217.9.......125..4...........8..671.......9.287....64..86.4....5",
    "1829" :    "...2..8.48......2..461..5..37...6...6...5...3...3...91..3..176..5......27.4..2...",
    "1902" :    "..6.39.4.1248.5.3...3..4..87.......5....1....2.......44..6..2...1.4.8573.5.39.4..",
    "1919" :    "..4..1.75...759........2..3632.7..5...........7..8.4924..5........148...78.3..2..",
    "2039" :    "1.....4.3.6.59.7...48....9.....87.....29.35.....26.....5....34...1.72.6.4.9.....1",
    "2058" :    ".89..74..153........4.8..5.....71.4..3..2..1..1.96.....6..9.1........572..81..36.",
    "2080" :    ".7......8..1.6......6..1243.5....4.99..4.3..64.8....3.2143..7......1.8..6......9.",
    "2110" :    ".92.1......674....41.6..7...6......534.....269......4...9..8.31....612......9.48.",
    "2114" :    "...3..82..4.....91.28..6.....4.6...8..61.37..7...4.1.....6..38.89.....5..32..9...",
    "2148" :    ".5..6.....419.2........8..7.67.3584...........8269.57.1..2........7.623.....8..5.",
    "2298" :    ".2...13.5.....48..5163........6..2.8...1.7...2.7..9........5917..54.....1.89...5.",
    "2432" :    "9.6...3....2.4....7...9..423576....9.........1....548761..7...4....3.9....3...7.5",
    "2631" :    "..4.1..23..7....9.....5...16.5..3.82.2.....5.79.8..1.43...8.....1....9..97..4.8..",
    "2839" :    "..2.5413.53.7....8...6....7.......82..7...3..68.......2....5...4....3.76.7329.5..",
    "3076" :    ".3.24.6......594.2.....853..93.2.....6.....1.....8.25..824.....4.789......9.37.2.",
    "3160" :    "..7..32...38..6..4...9..5.6.4....9.1.........8.9....4.7.1..8...5..2..38...47..6..",
    "3426" :    ".7.29.5.6..6..5...4..6....2.45..3.6....467....1.9..43.8....9..1...3..7..1.9.86.5.",
    "3535" :    "59.6........2.5....6.....84...9...51.78...62.92...8...28.....3....3.6........1.46",
    "3545" :    "...1.4...7..8.....192.......8.5.19.7..5.7.4..3.49.2.1.......376.....9..2...7.6...",
    "3594" :    "2.8....1....9.1..7.6..7.4....3...641.94...72.681...3....6.4..7.9..2.3....4....1.9",
    "3635" :    ".3.1.7.......8.6..91..........62..9865.9.3.4249..15..........23..1.6.......7.2.6.",
    "3684" :    "3..598...6....4..8..17...4.46......5..86.32..9......61.5...14..8..9....2...385..6",
    "3708" :    "5.9..6..4.7...5.....28....3.5..81397.........89357..2.7....81.....6...8.3..7..9.5",
    "3712" :    "..96...3....2..1.8.5...942.871........6...7........365.875...1.1.3..8....9...38..",
    "3867" :    ".....2..457..8...9.9.3.57....45.....3..8.7..1.....62....59.1.6.2...4..956..2.....",
    "3909" :    "7...491...2...1.79.4.8...6..3.9.8...4.......2...3.4.5..1...5.8.98.4...2...578...1",
    "3919" :    ".....38...4..5..935.8..9...3......8.194...367.2......5...7..6.291..4..3...25.....",
    "3944" :    ".....5..63.9.4.5.8...9....34..6...5.95.4.1.82.6...8..71....9...5.6.8.9.48..3.....",
    "4051" :    "..73......8...6...3..15..64.245....8.5..2..3.8....725.64..75..2...6...1......39..",
    "4092" :    ".82....4..1.83.9..7....45..8....51......7......42....7..64....5..3.16.9..4....31.",
    "4237" :    "..8...4..15..6..8.....3.12.4..5..26...62.49...72..6..4.24.1.....1..4..92..9...5..",
    "4238" :    "8..6..3......8..7.92....6...631.4..84.......52..5.693...2....96.8..2......1..8..3",
    "4296" :    "......3....1.....7.94.3.6...571..9..3..925..6..9..348...6.7.23.1.....8....3......",
    "4300" :    "..........8.924...2...3.98...12.96..6.5...8.2..98.71...97.8...3...312.4..........",
    "4336" :    "..9.1.57..16.49..2.....3..1....8.1..5.......7..8.3....4..3.....3..26.79..82.9.3..",
    "4359" :    "26...4.315......2......7..8..64....3..92.31..4....96..9..8......1......283.7...45",
    "4373" :    "825.1.7..4.6...9...........13...2.4....9.8....8.7...52...........9...3.6..3.9.415",
    "4434" :    "624......8.794.....5..3......8.2.5.3..1...7..5.3.7.6......8..6.....169.5......241",
    "4481" :    "9.8..7....2.3....6.7...294.593........1...6........215.328...9.1....6.2....1..5.4",
    "4571" :    "9..71....31..5.9...8....1.7.98..7.2....298....2.5..86.8.3....1...9.6..38....83..5",
    "4676" :    "75...2........73.5.6.18.2...7....1....28.14....4....7...9.14.8.2.79........7...42",
    "4707" :    "..3....14.69......4...1.83....5.6.41.9.1.7.8.62.3.4....12.6...7......92.74....1..",
    "4716" :    "..45.92..5.....9.8.7...2.43.4...8....8..1..9....2...8.63.1...5.1.8.....6..28.37..",
    "4762" :    "6....5.....94.7.8...7..24..2.5....49.7.....1.48....3.5..16..5...5.2.87.....5....4",
    "4796" :    ".9.6..2.5.354.....6..589.....8.......63...81.......9.....874..1.....675.4.9..5.2.",
    "4814" :    "....15..3..54....7..9.6.15........6.412...738.8........26.5.3..8....19..9..68....",
    "4825" :    "..14.8....3.....9...2..6.73...9.16.2..8...1..1.42.3...81.7..3...5.....1....8.92..",
    "4894" :    ".21.8..35.....3.41.5....92....5..7...4.....5...9..4....74....1.83.1.....21..7.58.",
    "5006" :    "...5.2.1.431......5..3.....91..65.....7...1.....41..36.....6..2......978.2.8.3...",
    "5007" :    "..1964..3......1.....5.86...45.8..1.8..2.1..9.7..5.26...83.6.....6......3..1978..",
    "5146" :    "2....79...586..3......2.85....3541.............3172....84.9......7..641...17....2",
    "5237" :    "...38..6.32.....8.8.6...3.243.97..1.....5.....1..23.496.5...7.4.9.....26.7..96...",
    "5333" :    "4..7.65..1.7..5...36......7....59.74..6...8..74.61....9......45...5..7.6..18.4..2",
    "5372" :    "..96..4..1.4.59..8...4..9..4.....237.2.....4.761.....9..2..3...8..19.7.2..7..68..",
    "5529" :    "....8.......4..1.925.6.37...251..89..3.....2..48..956...28.6.458.4..2.......1....",
    "5591" :    ".....2.856....81.3...75.9..57..9.....3.....5.....3..29..7.63...3.58....446.1.....",
    "5592" :    "7......6.93.4.......5.8674..4..69....9.....5....73..9..5327.6.......4.15.1......7",
    "5750" :    ".61...45.....6....32....8..4...21..7.7.9.5.1.8..64...5..9....26....1.....54...79.",
    "5802" :    ".83.15.2.6.....7.........1.7....2.6..2.1.8.3..5.3....7.9.........5.....1.1.92.37.",
    "5881" :    "3.1..59..4........5.9.4.2.8....96.3....8.4....6.35....9.8.1.7.3........1..25..6.9",
    "5885" :    "..5.........1.4.6242.39........2.3.11..8.7..52.7.4........53.9838.9.6.........6..",
    "6027" :    "1.....3..6..8.7.2..382...5.4...73....5.....9....48...7.6...247..7.6.8..3..4.....2",
    "6059" :    "4....7...7.5..8..4162...5..58...6.....68.37.....5...89..7...3922..1..8.7...7....1",
    "6084" :    "26..3.5..9............5..234....71....53.18....74....957..9............2..8.1..47",
    "6129" :    "79..........956.....6.13..293..61...5.8...9.6...52..132..47.1.....698..........69",
    "6162" :    "8...5...9...1..526...6..17..57....1....7.3....3....49..25..8...189..5...4...7...5",
    "6174" :    "...42.5..4.....18...5.8..2......369.9.......8.412......6..3.7...12.....5..8.75...",
    "6205" :    "17.25...99....8.37....9......7....6.5...6...8.3....9......1....28.5....46...27.85",
    "6250" :    ".5.42.....9..1.2.47.4..5..9...24.5.8.........5.1.83...9..1..3.54.5.6..7.....58.2.",
    "6317" :    ".3......886..24..9.79.8......7..2..4.4.....9.2..6..7......9.83.6..35..415......2.",
    "6626" :    ".....1.2......973.6.423.8..43.62.................75.81..6.435.2.571......4.7.....",
    "6776" :    "4......21..5.1...7..1482....42.9..5.5.......8.3..7.96....9652..9...3.8..21......5",
    "6787" :    "56.9..3......3.29...2...4.6...67....8.54.37.9....89...4.8...9...97.1......6..4.27",
    "6794" :    "..2.1....3....9.81.7..5.....3712.5...2.....6...5.3487.....8..5.26.9....7....6.2..",
    "6965" :    "26...8..99...53..8...1.2.6..27.....5..6...2..5.....67..3.5.1...6..38...11..4...92",
    "6979" :    ".....4.3.5.....9.89.7.21...39...21.7..6...3..7.13...52...41.2.51.9.....3.4.5.....",
    "7016" :    "..4712........3.176..49.28.7......5..5.....9..6......2.36.47..151.3........1598..",
    "7073" :    ".2...8...6.....891.....15.48..3.....39.1.5.72.....9..37.39.....541.....9...6...4.",
    "7078" :    "..2.4.8...6..7.5....81....7.2..8..7.3..7.9..6.9..6..1.8....47....4.3..6...7.2.4..",
    "7131" :    "...6.3.....592...7.....56387..5...23.4.....6.53...7..46742.....8...365.....4.9...",
    "7372" :    ".3.65...854....69789...4.....4.1....2..3.7..5....8.1.....4...26427....893...28.4.",
    "7442" :    "8.....1....6.5.7...1..9..52..7..8.91.8.....7.96.4..5..74..1..6...9.2.8....2.....3",
    "7455" :    "6.....1....2.5......79.46...7..6..4.9.35.78.1.4..2..5...51.67......3.4....1.....8",
    "7460" :    "...6...92218.....4..9........3.942..92.....87..127.5........9..7.....62183...6...",
    "7466" :    "........723..8..4.8...6..9...2....7..697.453..5....2...7..3...6.8..7..196........",
    "7482" :    ".....37..5.4.2......8...1268..3.......74.69.......5..1925...8......6.2.5..69.....",
    "7692" :    "9.65.......2.8...55....2.47...6...94....5....81...7...38.1....66...2.7.......58.9",
    "7726" :    "..5....1....142..62....68..6.34......8.....2......83.7..12....59..531....2....6..",
    "7729" :    "9....3.4...82.4.7.3.6.1.......4..7.9.9.....6.8.5..7.......4.2.7.7.5.94...5.1....8",
    "7799" :    "..3.1..5.1..57.....5...83.6.3..45....15...94....18..7.4.68...3.....62..7.8..5.1..",
    "7806" :    "..2...45...814...33.1.59.....3......7..3.5..2......1.....71.9.64...283...26...8..",
    "8385" :    ".573.....2...68..4..3.7.....35...9.8.4.6.7.5.9.6...24.....8.4..1..73...2.....451.",
    "8442" :    ".......65.1....9....269...7.43.71.8....2.3....2.86.71.3...274....7....5.26.......",
    "8522" :    "...72..3.2.3.5..97...1.345.......319....7....316.......329.4...45..1.6.3.8..36...",
    "8616" :    ".5.....873..2...5..8..53.....54...72.7.6.2.4.21...59.....89..6..2...1..886.....1.",
    "8727" :    "...1...5...47.2..88..4..927..1..3.4...........3.8..7..192..4..35..3.18...7...6...",
    "8779" :    "3.6..4....7...9.....1...7239..4..8..863...491..7..8..5258...9.....8...6....7..3.4",
    "8854" :    "..4..6..........64....91.2395.3...8.1.8...3.2.2...9.4726.81....78..........7..9..",
    "8882" :    "89..34...7.5....32...8.....4.6.72....2.....9....45.2.8.....8...37....5.6...31..47",
    "8931" :    "75.3..2...3....6.9....76.....3..75..6..2.9..1..26..8.....45....3.7....4...8..2.63",
    "8980" :    ".548....2...2..16.3...4.9..615.7.......6.2.......3.796..9.6...3.83..9...4....785.",
    "9015" :    "54....1...7..542..6.9..2..546..8.......4.6.......1..543..6..5.9..437..6...6....71",
    "9125" :    "86....2.5.1..2..9....7.3......9..8..746...932..9..2......3.8....2..9..5.5.4....68",
    "9189" :    ".24..9......124..9.7.6.........3.57.74.....21.81.4.........7.3.3..265......9..28.",
    "9231" :    "23...6..8...2...6..78..5.434.1....8.3...4...2.8....4.512.8..39..9...4...8..9...76",
    "9287" :    ".49..82.....19..861....54...6...2...9.......3...7...2...24....585..26.....65..87.",
    "9600" :    "62..54.....3....575.....63.....9.5..764...291..1.7.....16.....423....1.....16..85",
    "9619" :    "4.......22...86....93..2..19.14..72...........82..96.37..8..31....26...88.......4",
    "9642" :    "..8.5..4.5..3..1.7....79.2..9.....5...72.56...3.....1..4.96....8.2..1..4.7..4.8..",
    "9663" :    "3..7.......7..52.1548.......561....3...3.8...4....978.......3156.39..4.......2..9",
    "9713" :    ".....7....3.....4.86.5.3..7..5.9..742.6...1.919..7.5..9..4.2.53.5.....1....6.....",
    "9991" :    "..62..51..8.5..2.345.....7......389...........194......4.....596.5..9.2..71..26..",
    "11262" :   ".38..7.4...5..8...14...2...5.7....1..6..1..2..2....9.7...5...98...4..6...9.6..57.",
    "11316" :   "..7..1...5..8....9.94.53....35......9.2.6.1.7......68....23.96.4....8..3...4..7..",
    "12049" :   ".1..46.9..3....84...7......7.5..19.....6.8.....25..4.1......3...91....7..2.13..8.",
    "13728" :   "1..67..9......5...7....12...93...78...71.89...51...32...49....3...3......3..46..9",
    "15430" :   "..1....24..7.13.6.....4.75..3.5.......24.65.......9.4..89.6.....7.89.4..31....2..",
    "15700" :   ".8..56..2..5....962...3.4...9.2.....4..5.3..7.....4.3...9.6...875....9..6..18..5.",
    "15786" :   "....94...6..5.....1..72...6..3.7.68.52.....91.68.3.7..3...62..4.....8..7...41....",
    "16374" :   "58..3.7....7.9......1..4.2......92.6.16...48.9.28......5.2..9......7.3....8.5..62",
    "20604" :   "..435.7...6.8..3.9...9...6..42.....6...5.3...1.....92..2...4...7.3..5.4...1.986..",
    "21012" :   "2..6...5...1..5...64.2...3.7...68....3.....9....14...7.1...9.63...8..7...2...4..5",
    "21232" :   "...6.2.4.3.4.1....726...38....24...9..7...2..8...69....78...523....7.4.8.9.8.5...",
    "23406" :   "..5.....68.2...1.....62.8...512.....9..341..5.....971...8.93.....9...6.42.....3..",
    "23708" :   "6..4..52......839.5....9...7.9....8....713....2....4.7...8....4.421......57..6..2",
    "25855" :   "2....4.6.7.......3....12.8.6954.....1..6.8..2.....7946.6.28....9.......7.5.7....1",
    "26423" :   "3.8.4.2.......53......2.9...8...7.9.5.94.37.6.4.9...5...4.7......65.......7.8.5.4",
    "27959" :   "...69...2.2...3.67....5.4...1956..4.8.......6.6..3217...7.2....49.8...2.2...45...",
    "28454" :   "..78..519...2.....8.3.75.....6.5...8.5.....3.3...1.9.....19.6.3.....7...674..21..",
    "28523" :   ".2.1...6.......2.516..8..39..4.7...1...526...8...1.3..68..5..939.5.......7...1.8.",
    "29244" :   "17.....9.9.6..8..2....79....8.5.2...35.....14...3.4.5....73....8..9..4.6.1.....29",
    "30848" :   ".9...6..2..652....32.7.9....3.....7...23.19...1.....6....9.5.87....435..5..1...2.",
    "31226" :   ".3..5.6.9.6.7........2...4131....4....59.71....8....7397...4........8.3.8.1.9..6.",
    "31699" :   "...1.859....72.6.3...9....17......25.6.4.7.1.34......72....9...4.9.53....368.4...",
    "37332" :   "5...3.1.....7..8..3..154..79..5......4.....9......9..58..247..6..9..3.....7.6...2",
    "39410" :   "..3...52..6....1..8...21..7....3.2..31.....59..9.7....4..58...6..5....7..71...4..",
    "39611" :   "..8....96.....2...5..6..32.9..1.6.5.3.......8.6.7.4..2.59..1..4...8.....47....2..",
    "39647" :   "....1..5...7.259..56.9......7....6.1295...7846.3....9......3.17..689.4...5..4....",
    "40371" :   "49.7...31.....1..2...8.9.4...3....9...94.85...2....7...6.3.5...9..1.....35...4.79",
    "40951" :   "1.7...5..5...79.1.....1..249...3..5..72...36..6..4...123..6.....5.18...3..1...2.5",
    "43712" :   ".63.527....96.........3.42......3.5.48.....13.3.8......15.2.........61....678.34.",
    "44462" :   "9....3..2.8..74..6....6.1.55....8.4....237....1.4....32.5.4....3..68..1.8..3....4",
    "45010" :   "7..9.2.3.3.4...5....18.4..7.75.4....4.......8....9.17.1..6.37....7...8.3.3.1.5..2",
    "45573" :   "..7.2...184.1......213...8.....45..3.6.....4.2..76.....1...736......6.785...3.1..",
    "48190" :   "3.6.7...91......6..5....873.....2....319.824....7.....915....3..8......52...5.9.4",
    "50395" :   ".1...63.........7.6.35...19...8.3..7..2.6.9..5..7.1...14...87.6.2.........72...3.",
    "51535" :   "..3...7.6...927..32......5.....5..82...6.3...87..4.....2......93..471...6.1...5..",
    "51559" :   "872.3.4....1....5.45...92.....79..4...........1..58.....65...81.2....6....8.6.324",
    "54258" :   ".1..5...792....6.3.4...21.....8.19......4......57.3.....43...8.6.1....792...7..6.",
    "54668" :   "..9....8.82.....67.64.8.91......38..3..6.8..9..82......86.5.29.49.....76.7....5..",
    "55271" :   ".8.....6..2.3..1.56..8.1...7...146..29.....14..869...2...9.2..19.2..8.7..7.....3.",
    "57194" :   "79..4........31.2..3.2..9....3.785...28...14...461.8....9..4.7..5.82........5..38",
    "60892" :   "...5.7..82..1..6...1...6.92..6....1.7.......3.8....2..13.6...4...9..3..18..7.2...",
    "61541" :   "...57...9.....98.38.9...56......29.6.9.....8.6.39......68...1.49.46.....2...85...",
    "62621" :   ".......643.5.26.......189.5...7....8.97...51.4....1...5.124.......56.1.723.......",
    "62875" :   "..41..76..2..8..1..5...398...2..6.5...........6.7..2...162...7..8..6..4..93..78..",
    "68741" :   "4..25.....3...4.565.63..4..9.1.......4..8..2.......6.1..2..57.469.1...8.....37..2",
    "69321" :   "1.8..49.......7.2..6...8.7..3.45....48.....61....13.4..9.7...5..4.2.......76..4.9",
    "70861" :   ".46.3...93.2...87.7......4....75.....1.4.3.8.....29....2......1.53...9.81...4.35.",
    "71382" :   ".27..8...14.5.....5.....416.7..6.5.3.........3.1.9..4.968.....4.....7.62...6..93.",
    "71489" :   "..36..5........2.9.6.57..3....8..65...4...8...86..4....9..87.4.7.1........2..19..",
    "74642" :   ".375..9.84.....2.5..5..1.......1.694..9...5..184.9.......1..4..3.1.....25.8..217.",
    "78086" :   ".....71.....98.76....436..575....4.3..2...5..3.6....791..649....45.72.....83.....",
    "78855" :   "..5.96..14...3.......4...8.1.2...4.6.6..5..7.3.7...1.5.2...9.......2...85..76.3..",
    "79862" :   "8.1..4....4.9.26........5.44.9.....6..56832..3.....7.59.7........41.9.7....2..9.3",
    "83958" :   "..1....3.7...3.4.2..91...8..2.69.1....6...9....8.52.7..5...68..6.2.7...3.1....2..",
    "84395" :   ".....34...8.5.4..62...7.8...7.2...156.......484...1.9...8.9...23..8.6.4...94.....",
    "85009" :   ".3...1...95.872....7..9....5.3..4.19...9.8...79.6..4.2....6..5....485.73...1...4.",
    "86275" :   "..6....24...7.1..6.....6..8.7.9..8.1..96.45..6.1..5.4.2..4.....3..2.8...49....7..",
    "88242" :   "..4.....2.8..65...6....4..7.....716..9..8..7..753.....4..9....8...51..9.9.....7..",
    "89433" :   "46.....5...1..4..75..1......3.2...1...7...8...5...7.2......9..88..4..2...1.....73",
    "91230" :   "2.......3.6....5789..74.6......58...3..9.4..5...27......3.27..1876....9.4.......7",
    "91250" :   "....815.3.....6.7.6...9.18......264.9..8.4..7.863......63.2...1.4.6.....5.213....",
    "93020" :   ".29.8....3.6..9..4.7..1....158....7.6...7...1.3....865....6..1.5..1..9.3....3.64.",
    "94280" :   "....29..1...36.7..27.....9.41....9..39..7..28..5....64.2.....16..7.93...8..25....",
    "96990" :   "4.7...38..8..7...9....8..25....2157.....6.....9245....23..4....6...1..9..51...8.6",
    "99400" :   "..42...19.1..8.5..8...5.3......9.2...9.6.7.3...7.2......9.3...4..5.7..8.48...57..",
}