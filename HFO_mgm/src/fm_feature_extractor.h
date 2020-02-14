// -*-c++-*-

#ifndef FM_FEATURE_EXTRACTOR_H
#define FM_FEATURE_EXTRACTOR_H

#include <rcsc/player/player_agent.h>
#include "feature_extractor.h"
#include <vector>

/**
 * This feature extractor creates the high level feature set used by
 * Barrett et al.
 * (http://www.cs.utexas.edu/~sbarrett/publications/details-THESIS14-Barrett.html)
 * pages 159-160.
 */
class FMFeatureExtractor : public FeatureExtractor {
public:
  FMFeatureExtractor(int num_teammates, int num_opponents,
                            bool playing_offense);
  virtual ~FMFeatureExtractor();

  // Updated the state features stored in feature_vec
  virtual const std::vector<float>& ExtractFeatures(const rcsc::WorldModel& wm,
						    bool last_action_status);

  //override FeatureExtractor::valid
  //this method takes a pointer instead of a reference
  bool valid(const rcsc::PlayerObject* player);

protected:
  // Number of features for non-player objects.
  const static int num_basic_features = 12;
  // Number of features for each teammate and opponent in game.
  const static int features_per_teammate = 1;
  const static int features_per_opponent = 1;
};

#endif // FM_FEATURE_EXTRACTOR_H
