/*
- Funzione per generare un identificatore univoco:
1) Math.random() genera un numero pseudo-casuale tra 0 e 1
2) .toString(36) converte la parte decimale del numero appena generato in una stringa utilizzando la base 36.
    - La base 36 usa le cifre da 0 a 9 e rappresenta inoltre tutte le lettere da A a Z (con numeri da 10 a 35)
    - Per ottenere il numero convertito, si continua a moltiplicare la parte decimale *36 fino a quando o non si ottiene più
      una parte decimale significativa oppure qunado si raggiunge la precisione massima di Javascript per un numero in virgola mobile
      (max 16-17 cifre decimali)
    - Ad ogni moltiplicazione *36 viene presa la parte intera è convertita in base a come viene rappresentata in base 36
    - Alla fine si avrà quindi una stringa di questo tipo: 'h2wkrsqqhamzlop'
 4) .substr(2, 9) prende solamente la sottostringa che va dal terzo carattere e ne prende 9
 5) Quindi alla fine avremo un identificatore di questo tipo: 'wkrsqqham'
 Prob. di generare due session_id uguali: (1/36)^9
*/
export function generateUniqueSessionId(): string {
    return `session-${Math.random().toString(36).substr(2, 9)}`;
}