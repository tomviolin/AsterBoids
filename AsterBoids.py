#
#   Tom Hansen --  tomh@uwm.edu
#
#   Dec 6, 2017
#
#   I was experimenting with a boids flocking algorithm using the very
#   simply rendered shape of a narrow white hollow triangle pointed in the direction
#   of motion, drawn on a black background.  Suddenly the old arcade game
#   came to mind, thus was born AsterBoids.
#
#   I Googled and found at least two other people had thought of this mash-up
#   before me, so I can't take the credit for inventing it.
#
#   Note that I used some code from an example somewhere online
#   for a pygame particle system (original citation will be restored when I get a chance)
#   so most variables referring to boids are called "p" or "particle".  Once
#   I introduced bullets, I started using "b" for that so I decided to just
#   let "p" continue to represent "boid".
#
#   HOW TO PLAY:  (if you want to call it that...)
#
#   LEFT ARROW or SCROLL WHEEL DOWN:  Steer shooter left / counterclockwise
#   RIGHT ARROW or SCROLL WHEEL UP:   Steer shooter right / clockwise
#   UP ARROW or RIGHT MOUSE BUTTON:   activate thruster on shooter
#   SPACE BAR or LEFT MOUSE BUTTON:   FIRE!
#
#   When a boid is "shot" a new boid is born and comes in from the upper left corner of the screen.
#

# various imports
from __future__ import print_function, division
import pygame, pygame.gfxdraw, random
from sklearn.neighbors import NearestNeighbors
from math import *
from math import pi
import ctypes
from ctypes import windll

# --- various constants ---
BOIDS=650                   # how many boids
BOID_SIZE=10.               # base boid size
WINDOW_WIDTH=1920           # size of screen (horrible hard code sorry)
WINDOW_HEIGHT=1080          # size of screen (horrible hard code sorry)
HOODRANGE=100.              # how far does a boid's neighbor'hood extend?
FLOCK_WT=2 # .5             # relative weight of flocking (matching velocity) behavior
CROWD_WT=30. # 5.8          # relative weight of anti-crowding behavior
CROWD_SPACE=5.              # number of radii to keep away from neighbors
SHOOTER_WEIGHT = 0  # 18.5  # repelling from shooter (incomplete)
SHOOTER_SPACE=35.           # how far does shooter repel boids
COHESION_WT= 15.            # relative weight of cohesion (attracted to neighbors)
FLEE_WT = 4.                # relative weight of fleeing from bullets behavior
FLEE_SPACE=80.              # how far boids can see bullets
SPEED_LIMIT=800.            # maximum speed of boids
WALL =70.                   # thickness of wall force-field
WALL_FORCE = 33.0           # strength of wall force-field
RADIUS_AGE = 2000000        # how long before a boid grows? (never really used this)
KNN=5                       # now many neighbors to consider (has significant performance impact)



def newt(x1,y1,m1,x2,y2,m2):
    """ newton's law of gravity more or less
     I know this isn't a planetary model or anthing,
     and most of the time it is used as a repelling force.
     This equation just turns out to represent a nicely behaved conservative
     field that works well with objects of different masses.
     """
    # d2 = square of the distance between bodies
    d2 = (x2-x1)**2 + (y2-y1)**2
    if d2<1.:
        # if objects are too close we short-circuit to avoid huge anomalies
        return (0.,0.)
    # calculate magnitude of gravitational force
    gmag = m1*m2/d2
    # calculate distance between bodies
    d = sqrt(d2)
    # multiply by unit vector from body 1 to 2
    gx = (x2-x1)/d * gmag
    gy = (y2-y1)/d * gmag
    # that's it.
    return (gx,gy)




class Particle:
    """ The boids (a.k.a. particles) """
    def __init__(self, x, y, radius, speed, angle, colour, surface, swarm):
        """ a blatant mish-mash of self-determination and caller control. """
        self.x = x
        self.y = y
        angle = random.randint(0,360)/180*pi
        speed=SPEED_LIMIT/2.
        self.vx = sin(angle)*speed
        self.vy = -cos(angle)*speed
        self.radius = 6. if random.randint(1,10)>1 else 15.
        self.age = self.radius * RADIUS_AGE
        self.surface = surface
        self.colour = colour
        self.swarm = swarm

    def move(self):
        """ Update speed and position and enforce speed limit """
        # for constant change in position values.
        if (self is not self.swarm[0]): self.radius = self.age / RADIUS_AGE
        speed=sqrt(self.vx**2 + self.vy**2)
        if (speed > SPEED_LIMIT) and self.swarm[0] is not self:
            self.vx /= speed / SPEED_LIMIT
            self.vy /= speed / SPEED_LIMIT
        if (speed < SPEED_LIMIT / 1.6) and self.swarm[0] is not self:
            self.vx /= 0.+speed / SPEED_LIMIT * 1.6
            self.vy /= 0.+speed / SPEED_LIMIT * 1.6

        # this mass-related arbitrary constant of 300 is problematic, but hey, it works.
        self.x += self.vx / 300
        self.y += self.vy / 300
        self.age = self.age + 1

    def draw(self):
        """ Draw the particle on screen"""
        angle=atan2(self.vy,self.vx)
        path = [ (self.x+cos(angle)*self.radius, self.y+sin(angle)*self.radius),
                (self.x+cos(angle-2.5)*self.radius,self.y+sin(angle-2.5)*self.radius),
                (self.x+cos(angle+pi)*self.radius*.6, self.y+sin(angle+pi)*self.radius*.4),
                (self.x+cos(angle+2.5)*self.radius,self.y+sin(angle+2.5)*self.radius)
        ]
        #pygame.gfxdraw.aacircle(self.surface,int(self.x),int(self.y),self.radius,self.colour)
        pygame.draw.aalines(self.surface, self.colour,True,path,1)
        #pygame.draw.aacircle(self.surface,self.colour,True, [(self.x,self.y),(self.x+1.0,self.y+1.0)],1)

    def bounce(self, myindex, distances, indices):
        """ this method originally just "bounced" the "particles" off the "walls."
        Now it handles the flocking behaviors.  But it's still called "bounce."  Fork me if you
        want to change it ;)
        """

        # start the force total at zero
        forcex=0.
        forcey=0.

        # compute forces exerted by walls (if any)
        if self.x < WALL:
            forcex += WALL_FORCE*(WALL-self.x)

        if self.x > self.surface.get_width() - WALL:
            forcex -= WALL_FORCE * (self.x - self.surface.get_width()+WALL)

        if self.y > self.surface.get_height() - WALL:  # bottom
            forcey -= WALL_FORCE * (self.y - self.surface.get_height()+WALL)

        if self.y < WALL:  # top
            forcey += WALL_FORCE * (WALL-self.y)

        # flocking behaviors

        # determine last nearest neighbor within range
        # (note we start at 1 because zero is always identity (itself)
        nnr = 1
        for i in range(1,len(distances)):
            if distances[i] <= HOODRANGE:
                nnr=i+1

        # if there's at least one neighbor and we aren't the shooter (index 0)...
        if nnr >= 2 and myindex > 0:
            # BOID RULE:
            # move towards average velocity of all neighbors
            totvx = 0
            totvy = 0
            totm = 0
            for i in indices[1:nnr]:
                if i > 0:
                    m = self.swarm[i].radius ** 2
                    totvx += self.swarm[i].vx * m
                    totvy += self.swarm[i].vy * m
                    totm += m
            if totm > 0:
                totvx /= totm
                totvy /= totm
                forcex += (totvx-self.vx)*FLOCK_WT
                forcey += (totvy-self.vy)*FLOCK_WT

            # BOID RULE:
            # avoid crowding
            for i in range(0,nnr):
                if distances[i] < self.radius*CROWD_SPACE:
                    weight = CROWD_WT
                    space=  CROWD_SPACE
                    if indices[i]==0:
                        weight = SHOOTER_WEIGHT
                        space = SHOOTER_SPACE
                    n = self.swarm[indices[i]]
                    # move away from this neighbor
                    (gx,gy) = newt(self.x,self.y,self.radius**2, n.x,n.y,n.radius**2)
                    forcex -= gx*weight
                    forcey -= gy*weight

            # BOID RULE:
            # cohesion
            for i in indices[1:nnr]:
                if i > 0:
                    gx,gy = newt(self.x,self.y,self.radius**2,self.swarm[i].x,self.swarm[i].y,self.swarm[i].radius**2)
                    forcex += gx * COHESION_WT
                    forcey += gy * COHESION_WT

        # f = ma =>  a = f/m  =>   du/dt = f/m
        self.vx += forcex / self.radius**2 * 10
        self.vy += forcey / self.radius**2 * 10

class Bullet:
    """ really not much more than a struct, perhaps my pre-OO age is showing...? """
    def __init__(self,swarm):
        self.swarm = swarm
        angle = atan2(swarm[0].vy,swarm[0].vx)
        speed = sqrt(swarm[0].vx**2 + swarm[0].vy**2)
        speed = speed + SPEED_LIMIT*2
        self.vy = sin(angle)*speed
        self.vx = cos(angle)*speed
        self.x = swarm[0].x + cos(angle)*swarm[0].radius
        self.y = swarm[0].y + sin(angle)*swarm[0].radius
        self.exploding=False
        self.explodesize = 0


def main():
    white = (255, 255, 255)
    black = (0,0,0)
    grey = (128,128,128)

    # -- WINDOWS ONLY --
    # for high-res displays with a DPI > 100%, we have to tell windows we are "DPI Aware" or the
    # system picks a lower resolution and stretches the display.  Isn't that nice?
    # ( I haven't tried this on Linux yet since I've been away from my Linux box (cry).
    # I'll be back to it very soon and I'll update this accordingly when I have time)
    ctypes.windll.user32.SetProcessDPIAware()
    true_res = (windll.user32.GetSystemMetrics(0), windll.user32.GetSystemMetrics(1))

    # These two lines make the sounds work properly on my system.  YMMV.
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.mixer.init()

    # init pygame.  (is this comment even necessary?)
    pygame.init()

    # create screen.  On my system weird things happened when I supplied (0,0) like you're
    # supposed to for full screen.  I just went ahead and hardcoded my display resolution
    # and decided to fix it later.  Later hasn't arrived yet, apparently.
    screen = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT), pygame.DOUBLEBUF | pygame.FULLSCREEN | pygame.HWSURFACE)

    # yea, these things never returned the real screen size unless I hardcoded above.  Later...
    xmax=screen.get_width()
    ymax=screen.get_height()

    # create clock object for use later in the game.
    clock = pygame.time.Clock()

    # start with no boids.
    particles = []

    # load the sounds and set relative volumes when necessary
    #   (nearly woke the gf and kids with those first two, damn...
    #    the other ones are much quieter)
    boom=pygame.mixer.Sound("BangMedium.wav")
    boom.set_volume(0.3)
    fire=pygame.mixer.Sound("fire.wav")
    fire.set_volume(0.2)
    thrust=pygame.mixer.Sound("thrust.wav")
    beat1 = pygame.mixer.Sound("beat1.wav")
    beat2 = pygame.mixer.Sound("beat2.wav")

    # create the boids!
    for i in range(BOIDS):
        # I'm not British but that particle example had this in it so... as usual I kept it.
        colour = [ random.randint(0,255) for i in range(3) ]
        x = random.randint(0, screen.get_width())
        y = random.randint(0, screen.get_height())
        speed = SPEED_LIMIT/2.
        angle = random.randint(0,360)/180*pi
        radius = 5
        # I kept the speed/angle thing because it probably produces a better random distribution
        # than a cartesian-based random sample because there's no corners.  Doesn't sound very scientific,
        # but there you have it.
        particles.append( Particle(x, y, radius, speed, angle, colour, screen, particles) )

    # the shooter.  This probably should have been
    # made a separate object by now because virtually
    # everything has a if-statement to handle index zero of the particles (boids) being
    # the shooter.  Later...
    particles[0].colour=white
    particles[0].radius = BOID_SIZE+7
    particles[0].x = screen.get_width()/2
    particles[0].y = screen.get_height()/2
    particles[0].vx = 0.01
    particles[0].vy = 0.01

    # oh yea and while we're at it let's create an empty array of bullets so we can
    # shoot at stuff.
    bullets=[]

    # we better not be done already!
    done = False

    # keep track of the beat/beep we're on
    beats=0
    while not done:

        # background beeps
        # I could do more with this, later.
        beats += 1;
        if beats == 30: beat1.play()
        if beats==60:
            beat2.play()
            beats=0

        # keep track of various events
        firenow = False
        leftnow = False
        rightnow = False
        upnow = False
        downnow = False
        # record event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key==ord('q'):
                    done = True
                    break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    rightnow=True
                elif event.button==5:
                    leftnow=True
                elif event.button == 1:
                    firenow=True
                elif event.button==3:
                    upnow=True

        keys = pygame.key.get_pressed()
        mbuts = pygame.mouse.get_pressed()
        if mbuts[2]: upnow=True
        if (keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or leftnow or rightnow):
            magmove=.07
            if leftnow or rightnow: magmove=.2
            angle=atan2(particles[0].vy,particles[0].vx)
            speed=sqrt(particles[0].vx**2 + particles[0].vy**2)
            angle += magmove if keys[pygame.K_RIGHT] or rightnow else -magmove
            particles[0].vy = sin(angle)*speed
            particles[0].vx = cos(angle)*speed
        if keys[pygame.K_UP] or keys[pygame.K_DOWN] or upnow or downnow:
            angle=atan2(particles[0].vy,particles[0].vx)
            speed=sqrt(particles[0].vx**2 + particles[0].vy**2)
            if keys[pygame.K_UP] or upnow: 
                speed += SPEED_LIMIT/10
                thrust.play()
            if keys[pygame.K_DOWN] or downnow:
                speed -= SPEED_LIMIT/10
                if speed <= 0: speed=.01
            particles[0].vy = sin(angle)*speed
            particles[0].vx = cos(angle)*speed
        else:
            particles[0].vx *= 0.95
            particles[0].vy *= 0.95
        if done:
            break

        screen.fill(black)
        positions = [ [p.x,p.y] for p in particles ]
        nbrs = NearestNeighbors(n_neighbors=(KNN if BOIDS>=KNN else BOIDS), algorithm='kd_tree',radius=0.5).fit(positions)
        distances,indices = nbrs.kneighbors(positions)
        for pidx in range(len(particles)):
            p = particles[pidx]
            p.move()
            p.bounce(pidx, distances[pidx],indices[pidx])
            p.draw()

        for b in bullets:
            extra=0
            if b.exploding:
                b.vx=0
                b.vy=0
                b.explodesize += 3
                if b.explodesize>20:
                    bullets.remove(b)
                    continue
                extra=b.explodesize
                pygame.draw.circle(screen, (255,0,0), (int(b.x),int(b.y)), 4+extra, 0)
            b.x+=b.vx/300
            b.y+=b.vy/300
            pygame.gfxdraw.aacircle(screen,int(b.x),int(b.y),4+extra,(255,255,0))
            pygame.gfxdraw.aacircle(screen,int(b.x),int(b.y),3+int(extra/2),(255,128,0))
            b.vx += 0
            b.vy += 0

        if keys[pygame.K_SPACE] or mbuts[0] or firenow:
            for i in range(3): bullets.append(Bullet(particles))
            fire.play()

        if len(bullets)>0:
            bpos=[[b.x,b.y] for b in bullets]
            dis, ixs = nbrs.kneighbors(bpos)
            todelete=[]
            todelb=[]
            for bidx in range(len(bullets)):
                # b is current bullet
                b=bullets[bidx]
                if b.exploding: continue
                # identify nearest particle
                cpi=ixs[bidx][0]
                p=particles[cpi]
                # record distance to particle
                dst = dis[bidx][0]
                # pygame.gfxdraw.aacircle(screen, int(p.x),int(p.y),int(dst),(255,0,0))
                # if the nearst boid isn't the shooter (index 0), is within distance, and is still real
                if cpi > 0 and dis[bidx][0] < particles[cpi].radius and p not in todelete:
                    todelete.append(particles[cpi])
                    todelb.append(b)
                    boom.play()
                else:
                    # deflect nearest neigbor away (test flocking effectiveness)
                    for cpi in range(len(ixs[bidx])):
                        if ixs[bidx][cpi] > 0:
                            p = particles[ixs[bidx][cpi]]
                            pdst = dis[bidx][cpi]
                            if pdst <= FLEE_SPACE:
                                # boid sees bullet
                                # calculate speed of bullet
                                magbv = sqrt(b.vx **2 + b.vy ** 2)
                                # calculate unit vector in direction of bullet
                                unitbvx = b.vx / magbv
                                unitbvy = b.vy / magbv
                                # calculate distance from bullet to boid
                                magdst = sqrt((p.x-b.x)**2 +(p.y-b.y)**2)
                                # project where the bullet will be when it has travelled that distance
                                projbx = b.x + unitbvx * magdst
                                projby = b.y + unitbvy * magdst
                                # flee from that spot
                                gx,gy = newt(p.x,p.y,p.radius**2, projbx, projby, BOID_SIZE**2)
                                p.vx -= gx * FLEE_WT
                                p.vy -= gy * FLEE_WT
                                # also flee from where bullet is now
                                gx,gy = newt(p.x,p.y,p.radius**2, b.x,b.y, BOID_SIZE**2)
                                p.vx -= gx * FLEE_WT
                                p.vy -= gy * FLEE_WT

                if b.x < 0 or b.x > screen.get_width() or b.y<0 or b.y > screen.get_height():
                    todelb.append(b)
            for b in todelb:
                b.exploding = True
            for p in todelete:
                if (p in particles):
                    particles.remove(p)
                    particles.append(Particle(-1,-1,9,SPEED_LIMIT/2,pi/4,(0,255,0),screen,particles))




        # print (clock.tick(),end='\r')

        # print(1000/clock.tick(60))


        pygame.display.flip()
    pygame.quit()

if __name__ == "__main__":
    main()
