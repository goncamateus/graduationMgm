#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "highlevel_feature_extractor.h"
#include <rcsc/player/intercept_table.h>
#include <rcsc/common/server_param.h>
#include "strategy.h"


using namespace rcsc;

HighLevelFeatureExtractor::HighLevelFeatureExtractor(int num_teammates,
                                                     int num_opponents,
                                                     bool playing_offense) :
  FeatureExtractor(num_teammates, num_opponents, playing_offense)
{
  assert(numTeammates >= 0);
  assert(numOpponents >= 0);
  numFeatures = num_basic_features + features_per_teammate * numTeammates
      + features_per_opponent * numOpponents;
  // numFeatures+=2; // action status, stamina
  feature_vec.resize(numFeatures);
}

HighLevelFeatureExtractor::~HighLevelFeatureExtractor() {}

const std::vector<float>&
HighLevelFeatureExtractor::ExtractFeatures(const rcsc::WorldModel& wm,
					   bool last_action_status) {
  featIndx = 0;
  const ServerParam& SP = ServerParam::i();
  const SelfObject& self = wm.self();
  const Vector2D& self_pos = self.pos();
  const float self_ang = self.body().radian();
  const PlayerPtrCont& teammates = wm.teammatesFromSelf();
  const PlayerPtrCont& opponents = wm.opponentsFromSelf();
  float maxR = sqrtf(SP.pitchHalfLength() * SP.pitchHalfLength()
                     + SP.pitchWidth() * SP.pitchWidth());
  // features about self pos
  // Allow the agent to go 10% over the playfield in any direction
  float tolerance_x = .1 * SP.pitchHalfLength();
  float tolerance_y = .1 * SP.pitchHalfWidth();
  // Feature[0]: X-postion
  addFeature(self_pos.x);

  // Feature[1]: Y-Position
  addFeature(self_pos.y);

  // Feature[2]: Self Angle
  addFeature(self_ang);

  float r;
  float th;
  // Features about the ball
  Vector2D ball_pos = wm.ball().pos();
  angleDistToPoint(self_pos, ball_pos, th, r);
  // Feature[3] and [4]: (x,y) postition of the ball
  if (playingOffense) {
    addFeature(ball_pos.x);
  } else {
    addFeature(ball_pos.x);
  }
  addFeature(ball_pos.y);
  // Feature[5]: Able to kick
  addFeature(self.isKickable());

  // Features about distance to goal center
  Vector2D goalCenter(SP.pitchHalfLength(), 0);
  if (!playingOffense) {
    goalCenter.assign(-SP.pitchHalfLength(), 0);
  }
  angleDistToPoint(self_pos, goalCenter, th, r);
  // Feature[6]: Goal Center Distance
  addFeature(r);
  // Feature[7]: Angle to goal center
  addFeature(th);
  // Feature[8]: largest open goal angle
  addFeature(calcLargestGoalAngle(wm, self_pos));
  // Feature[9]: Dist to our closest opp
  if (numOpponents > 0) {
    calcClosestOpp(wm, self_pos, th, r);
    addFeature(r);
  } else {
    addFeature(FEAT_INVALID);
  }

  // Features[10 - 10+T]: teammate's open angle to goal
  int detected_teammates = 0;
  for (PlayerPtrCont::const_iterator it=teammates.begin(); it != teammates.end(); ++it) {
    const PlayerObject* teammate = *it;
    if (valid(teammate) && teammate->unum() > 0 && detected_teammates < numTeammates) {
      addFeature(calcLargestGoalAngle(wm, teammate->pos()));
      detected_teammates++;
    }
  }
  // Add zero features for any missing teammates
  for (int i=detected_teammates; i<numTeammates; ++i) {
    addFeature(FEAT_INVALID);
  }

  // Features[10+T - 10+2T]: teammates' dists to closest opps
  if (numOpponents > 0) {
    detected_teammates = 0;
    for (PlayerPtrCont::const_iterator it=teammates.begin(); it != teammates.end(); ++it) {
      const PlayerObject* teammate = *it;
      if (valid(teammate) && teammate->unum() > 0 && detected_teammates < numTeammates) {
        calcClosestOpp(wm, teammate->pos(), th, r);
        addFeature(r);
        detected_teammates++;
      }
    }
    // Add zero features for any missing teammates
    for (int i=detected_teammates; i<numTeammates; ++i) {
      addFeature(FEAT_INVALID);
    }
  } else { // If no opponents, add invalid features
    for (int i=0; i<numTeammates; ++i) {
      addFeature(FEAT_INVALID);
    }
  }

  // Features [10+2T - 10+3T]: open angle to teammates
  detected_teammates = 0;
  for (PlayerPtrCont::const_iterator it=teammates.begin(); it != teammates.end(); ++it) {
    const PlayerObject* teammate = *it;
    if (valid(teammate) && teammate->unum() > 0 && detected_teammates < numTeammates) {
      addFeature(calcLargestTeammateAngle(wm, self_pos, teammate->pos()));
      detected_teammates++;
    }
  }
  // Add zero features for any missing teammates
  for (int i=detected_teammates; i<numTeammates; ++i) {
    addFeature(FEAT_INVALID);
  }

  // Features [10+3T - 10+6T]: x, y, unum of teammates
  detected_teammates = 0;
  for (PlayerPtrCont::const_iterator it=teammates.begin(); it != teammates.end(); ++it) {
    const PlayerObject* teammate = *it;
    if (valid(teammate) && teammate->unum() > 0 && detected_teammates < numTeammates) {
      if (playingOffense) {
        addFeature(teammate->pos().x);
      } else {
        addFeature(teammate->pos().x);
      }
      addFeature(teammate->pos().y);
      addFeature(teammate->unum());
      detected_teammates++;
    }
  }
  // Add zero features for any missing teammates
  for (int i=detected_teammates; i<numTeammates; ++i) {
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
  }

  // Features [10+6T - 10+6T+3O]: x, y, unum of opponents
  int detected_opponents = 0;
  for (PlayerPtrCont::const_iterator it = opponents.begin(); it != opponents.end(); ++it) {
    const PlayerObject* opponent = *it;
    if (valid(opponent) && opponent->unum() > 0 && detected_opponents < numOpponents) {
      if (playingOffense) {
        addFeature(opponent->pos().x);
      } else {
        addFeature(opponent->pos().x);
      }
      addFeature(opponent->pos().y);
      addFeature(opponent->unum());
      detected_opponents++;
    }
  }
  // Add zero features for any missing opponents
  for (int i=detected_opponents; i<numOpponents; ++i) {
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
    addFeature(FEAT_INVALID);
  }

  const int self_min = wm.interceptTable()->selfReachCycle();
  const int mate_min = wm.interceptTable()->teammateReachCycle();
  const int opp_min = wm.interceptTable()->opponentReachCycle();

  bool isIntercept = false;
  int num = wm.self().unum();
  if ( ! wm.existKickableTeammate()
        && ( self_min <= 3
            || ( self_min <= mate_min
                  && self_min < opp_min + 3 )
            )
        )
    isIntercept = true;

Vector2D myInterceptPos = wm.ball().inertiaPoint( self_min );
Vector2D me = wm.self().pos();
Vector2D homePos = Strategy::i().getPosition( wm.self().unum() );
Vector2D ball = wm.ball().pos();

{
    if( num > 5 && ! wm.existKickableTeammate()
        && ( self_min <= 3
             || ( self_min < mate_min + 3
                  && self_min < opp_min + 4 ) 
              )
         )
    {
            isIntercept = true;
    }

    if( num > 6 && ( wm.existKickableOpponent() || (opp_min < 5 && mate_min > opp_min) ) &&
        ( ball.dist( homePos ) < 10.0 || (self_min < 6 && ball.dist( homePos ) < 15.0 ) ||
          (self_min < 6 && ball.dist( homePos ) < 25.0 && num > 8)  )
        && ball.x > -36.0 && wm.self().stamina() > 4000.0 && !wm.existKickableTeammate() )
    {
            isIntercept = true;
    }

}

{
    if ( num < 6 && ! wm.existKickableTeammate()
         && ( self_min <= 2
              || ( self_min <= mate_min // self_min < mate_min + 3 bud!
                   && self_min < opp_min + 2 ) // base: self_min < opp_min + 4
              )
         )
    {
        isIntercept = true;
    }

    if( (num == 4 || num == 5) && self_min <= 4 && ball.absY() > 15.0 && ball.x < -36.0 &&
        me.dist(Vector2D(-50.0,0.0)) < ball.dist(Vector2D(-50.0,0.0)) &&
        !wm.existKickableTeammate())
    {
        isIntercept = true;
    }
}

{
    if( ( num > 6 && mate_min > self_min + 1 && self_min < opp_min + 3 &&
          myInterceptPos.x > 15.0 ) ||
        ( num == 11 && mate_min > self_min && self_min < opp_min + 6 &&
          myInterceptPos.x > 5.0 && myInterceptPos.x > wm.offsideLineX() - 3 ) ||
        ( num > 6 && myInterceptPos.x > 20.0 && self_min < mate_min + 3 && opp_min < mate_min &&
          opp_min < self_min && !wm.existKickableTeammate() )
      )
    {
        isIntercept = true;
    }
}

{
    int goalCycles = 100;

    for( int z=1; z<15; z++ )
      if( wm.ball().inertiaPoint( z ).x < -52.5 && wm.ball().inertiaPoint( z ).absY() < 8.0 )
      {
         goalCycles = z;
         break;
      }

    if( (wm.self().unum() == 2 || wm.self().unum() == 3) && goalCycles != 100 &&
        mate_min >= goalCycles && me.x < -45.0 && !wm.existKickableTeammate() )
    {

      if( wm.ball().inertiaPoint( self_min ).x > -52.0 )
        isIntercept = true;
    }
}

  addFeature(isIntercept);
  addFeature(self.stamina());

  assert(featIndx == numFeatures);
  // checkFeatures();
  return feature_vec;
}

bool HighLevelFeatureExtractor::valid(const rcsc::PlayerObject* player) {
  if (!player) {return false;} //avoid segfaults
  const rcsc::Vector2D& pos = player->pos();
  if (!player->posValid()) {
    return false;
  }
  return pos.isValid();
}
