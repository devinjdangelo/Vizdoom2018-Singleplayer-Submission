#!/usr/bin/env python3

import vizdoom as vzd
from random import choice
import os
from model.Agent import DoomAgent
import skimage as skimage
from skimage import color, transform
from EvalArgs import args
import numpy as np
# Name your agent
#
# Note: The grading infrastructure will provide the expected agent name
# in the environment variable `CROWDAI_AGENT_NAME`
# If your agent does not use this environment variable then the score
# will not be counted against your crowdai user.
agent_name = "DoomGai"
server_agent_name = os.getenv("CROWDAI_AGENT_NAME", agent_name)

DEFAULT_WAD_FILE = "mock.wad"


def run_game():
    agent = DoomAgent(args)
    game = vzd.DoomGame()
    game.load_config("config/doom2.cfg")
    # and select color
    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    color = 0
    game.set_doom_scenario_path(DEFAULT_WAD_FILE)
    # This will be internally overriden in the grading environment with the relevant WAD file

    game.add_game_args("-join localhost +name {agent_name} +colorset {colorset}".format(
        agent_name=server_agent_name,
        colorset=color
    ))

    game.set_console_enabled(True)
    game.set_window_visible(False)

    game.init()

    # 5 buttons:
    # MOVE_FORWARD
    # MOVE_RIGHT
    # MOVE_LEFT
    # TURN_LEFT
    # TURN_RIGHT
    # ATTACK

    buttons_num = game.get_available_buttons_size()
    abuffer = np.zeros(buttons_num)
    # Play until the game (episode) is over.
    while not game.is_episode_finished():

        if game.is_player_dead():
            # Use this to respawn immediately after death, new state will be available.
            game.respawn_player()

            # Or observe the game until automatic respawn.
            # game.advance_action();
            # continue;
        
        # Analyze the state ... or not
        s = game.get_state()
        m, weapon = process_game_vars(s.game_variables)
        f = s.screen_buffer
        f = f[:-80,:,:] #crop out the HUD
        f = skimage.transform.resize(f,(128,128,3))
        f = skimage.color.rgb2lab(f)
        # print(s)
        vz_action,_,_,_ = agent.choose_action(f,m,abuffer,0,False,weapon)
        # Make your action.
        game.make_action(vz_action,4)
        abuffer = vz_action

        # Log time and frags/kills every ~10 seconds
        if s and s.number % 350 == 0:
            print("Time: {}, Kills: {}, Frags: {}",
                  game.get_episode_time(),
                  game.get_game_variable(vzd.GameVariable.KILLCOUNT),
                  game.get_game_variable(vzd.GameVariable.FRAGCOUNT))

    print("Episode FINISHED !")
    game.close()
    
def process_game_vars(m_raw):
    
    selected_weapon = m_raw[1]
    
    fist_active = 1 if selected_weapon==1 else 0
    pistol_active = 1 if selected_weapon==2 else 0
    shotgun_active = 1 if selected_weapon==3 else 0
    chaingun_active = 1 if selected_weapon==4 else 0
    rocket_active = 1 if selected_weapon==5 else 0
    plasma_active = 1 if selected_weapon==6 else 0
    bfg_active = 1 if selected_weapon==7 else 0
    super_active = 1 if selected_weapon==8 else 0
    chainsaw_active = 1 if selected_weapon==9 else 0

    #weap1 = 0 if fist only and =1 if fist and chainsaw
    has_fist = 1 if m_raw[10]>0 else 0
    #weap2 = 1 if pistol
    has_pistol = 1 if m_raw[11]>0 else 0
    #weap3 = 1 if shotty 
    has_shotgun = 1 if m_raw[12]>0 else 0
    #weap4 = 1 if supershotty
    has_chaingun = 1 if m_raw[13]>0 else 0
    has_rocket = 1 if m_raw[14]>0 else 0
    has_plasma = 1 if m_raw[15]>0 else 0
    has_bfg = 1 if m_raw[16]>0 else 0
    has_super = 1 if m_raw[17]>0 else 0
    has_chainsaw = 1 if m_raw[18]>0 else 0
    
    #ammo2 = pistol bullets
    ammo2 = m_raw[20]
    #ammo3 = shotgun shells
    ammo3 = m_raw[21]
    #ammo4 = rockets
    ammo4 = m_raw[22]
    #ammo5 = cells
    ammo5 = m_raw[23]

    health = m_raw[2]
    armor = m_raw[3]

      
            
    m = [fist_active,pistol_active,shotgun_active,chaingun_active,rocket_active,
         plasma_active,bfg_active,super_active,chainsaw_active,has_fist,has_pistol,has_shotgun,
         has_chaingun,has_rocket,has_plasma,has_bfg,has_super,has_chainsaw,
         ammo2,ammo3,ammo4,ammo5,health,armor,0,0,0,
         0,0]
   
    return m,selected_weapon

if __name__ == "__main__":
    """
    The `run_game` function plays a single DoomGame, and the submitted agent
    should continue to try to join and play a new game as long as the server
    doesnot actively kill it.
    """
    while True:
        print("Connecting to Game Episode....")
        run_game()
