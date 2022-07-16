/*
 * FILE: tty.cpp
 * AUTHOR: Sean Gallogly
 * CREATED: 07/01/2022
 *
 * IMPLEMENTATION NOTES: credit goes to PBS answer here:
 * https://stackoverflow.com/questions/29335758/using-kbhit-and-getch-on-linux 
 * and anon answer here:
 * https://stackoverflow.com/questions/421860/capture-characters-from-standard-input-without-waiting-for-enter-to-be-pressed
 *
 */

#include <sys/ioctl.h>
#include <signal.h>
#include <termios.h>
#include "ttyManip/tty.h"

void enable_raw_mode(FILE **fp)
{
	termios term; /* c struct with terminal bit flags */
	tcgetattr(fileno(*fp), &term); /* get term bit flags from tty pointed to by fp */
	term.c_lflag &= ~(ICANON | ECHO); /* block term's canonical mode and echo */
	tcsetattr(fileno(*fp), TCSANOW, &term); /* update fp's flags to updated term's flags immediately */
}

void disable_raw_mode(FILE **fp)
{
	termios term;
	tcgetattr(fileno(*fp), &term);
	term.c_lflag |= ICANON | ECHO; /* set term back to canonical mode and to echo */
	tcsetattr(fileno(*fp), TCSANOW, &term);
}

bool kbhit(FILE **fp)
{
	int byteswaiting;
	ioctl(fileno(*fp), FIONREAD, &byteswaiting); /* get num immediately avail bytes in fp */
	return byteswaiting > 0;
}

/*
 * IMPLEMENTATION NOTE: on call, the read could return nothing. This fnctn should 
 *                      only be called after a call to kbhit to ensure our tty has
 *                      something in it to actually read :>
 */
char getchar(FILE **fp)
{
	char buf = 0; /* single character buffer */
	struct termios old;
	if (tcgetattr(fileno(*fp), &old) < 0) perror("tcsetattr()");
	old.c_cc[VMIN] = 1; /* set min noncanonical read bytes to 1*/
	old.c_cc[VTIME] = 0; /* set read timeout to 0s */
	if (tcsetattr(fileno(*fp), TCSANOW, &old) < 0) perror("tcsetattr ICANON");
	if (read(fileno(*fp), &buf, 1) < 0) perror("read()"); /* perform the read */
	return buf; /* return the buffer */
}

int init_tty(FILE **fp)
{
	// attempt to open the controlling tty
	if ((*fp = fopen(ctermid(NULL), "r+")) == NULL) return -1;
	setbuf(*fp, NULL); /* set tty to be unbuffered */
	enable_raw_mode(fp); 
	return 0;
}

void process_input(FILE **fp, int tts, int trial_num)
{
	if (kbhit(fp))
	{
		int c = 0;
		switch (getchar(fp))
		{
			case 'p': /* pause case */
			    printf("[INFO] Simulation paused at time step %d on trial %d\n", tts, trial_num);
				while (true)
				{
					if (kbhit(fp))
					{
						c = getchar(fp);
						if (c == 'c')
						{
							printf("[INFO] Continuing...\n");
						 	break;
						}
					}
				}
				break;
			// TODO: add in other cases as needed
		}
	}
}

int reset_tty(FILE **fp)
{
	disable_raw_mode(fp); /* reset tty to canonical mode and echoing user input */
	tcflush(fileno(*fp), TCIFLUSH); /* flush input so characters don't show up later */
	fclose(*fp); /* close this stream */
	*fp = NULL; 
	return 0;
}
