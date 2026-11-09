Write a function `isValidCardNumber(input)` that checks a payment card number with the Luhn algorithm.

The space character and the hyphen `-` may appear anywhere in the input as separators and are ignored; any other character (a tab, a letter, a dot, a sign) makes the input invalid. After removing them, the number must consist only of digits and be 12 to 19 digits long, and its Luhn checksum must be valid: going from the rightmost digit to the left, double every second digit (the second from the right, the fourth, ...), subtract 9 from any doubled digit above 9, sum all digits, and the total must be a multiple of 10.

Return `true` or `false`. Anything that is not a string (numbers, `null`, `undefined`) gives `false`; never throw.
