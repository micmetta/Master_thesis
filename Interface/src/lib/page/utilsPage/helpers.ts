/*
- Function for generating a unique identifier:
1) Math.random() generates a pseudo-random number between 0 and 1
2) .toString(36) converts the decimal part of the number just generated into a string using base 36.
    - Base 36 uses the digits 0 to 9 and also represents all letters A to Z (with numbers 10 to 35)
    - To obtain the converted number, we continue to multiply the decimal part *36 until either we no longer obtain
      a significant decimal part or when the maximum precision of Javascript is reached for a floating-point number
      (max. 16-17 decimal digits)
    - At each multiplication *36 the integer part is taken and converted to base 36
    - At the end you will then have a string of this type: 'h2wkrsqqhamzlop'
 4) .substr(2, 9) takes only the substring that goes from the third character and takes 9
 5) So at the end we will have an identifier of this type: 'wkrsqqham'
 The Probability of generating two identical session_ids is: (1/36)^9
*/
export function generateUniqueSessionId(): string {
    return `session-${Math.random().toString(36).substr(2, 9)}`;
}