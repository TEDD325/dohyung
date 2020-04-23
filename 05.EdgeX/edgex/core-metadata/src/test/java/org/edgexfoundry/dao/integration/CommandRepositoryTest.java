/*******************************************************************************
 * Copyright 2016-2017 Dell Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *
 * @microservice: core-metadata
 * @author: Jim White, Dell
 * @version: 1.0.0
 *******************************************************************************/

package org.edgexfoundry.dao.integration;

import static org.edgexfoundry.test.data.CommandData.TEST_CMD_NAME;
import static org.edgexfoundry.test.data.CommandData.checkTestData;
import static org.edgexfoundry.test.data.CommandData.newTestInstance;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.List;

import org.edgexfoundry.Application;
import org.edgexfoundry.dao.CommandRepository;
import org.edgexfoundry.domain.meta.Command;
import org.edgexfoundry.test.category.RequiresMongoDB;
import org.edgexfoundry.test.category.RequiresSpring;
import org.edgexfoundry.test.category.RequiresWeb;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.SpringApplicationConfiguration;
import org.springframework.test.context.junit4.SpringJUnit4ClassRunner;
import org.springframework.test.context.web.WebAppConfiguration;

@RunWith(SpringJUnit4ClassRunner.class)
@SpringApplicationConfiguration(classes = Application.class)
@WebAppConfiguration("src/test/resources")
@Category({RequiresMongoDB.class, RequiresSpring.class, RequiresWeb.class})
public class CommandRepositoryTest {

  @Autowired
  private CommandRepository repos;
  private String id;

  @Before
  public void creatTestData() {
    Command cmd = newTestInstance();
    repos.save(cmd);
    id = cmd.getId();
    assertNotNull("new test Command has no identifier", id);
  }

  @After
  public void cleanup() {
    repos.deleteAll();
  }

  @Test
  public void testFindOne() {
    Command cmd = repos.findOne(id);
    assertNotNull("Find one returns no command", cmd);
    checkTestData(cmd, id);
  }

  @Test
  public void testFindOneWithBadId() {
    Command cmd = repos.findOne("foo");
    assertNull("Find one returns command with bad id", cmd);
  }

  @Test
  public void testFindAll() {
    List<Command> cmds = repos.findAll();
    assertEquals("Find all not returning a list with one addressable", 1, cmds.size());
    checkTestData(cmds.get(0), id);
  }

  @Test
  public void testFindByName() {
    List<Command> cmds = repos.findByName(TEST_CMD_NAME);
    assertEquals("Find by name returns no command", 1, cmds.size());
    checkTestData(cmds.get(0), id);
  }

  @Test
  public void testFindByNameWithBadName() {
    List<Command> cmd = repos.findByName("badname");
    assertTrue("Find by name returns command with bad name", cmd.isEmpty());
  }

}
